import tensorflow as tf
from tensorflow.keras import layers, models, Input, mixed_precision
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, precision_recall_curve, average_precision_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the DataFrame
file_path = r"C:\Users\meh91075\Downloads\AutoSplit_all_in_one_dataset_05112024.csv"

df = pd.read_csv(file_path)

# Entferne label_standard
df = df.drop('label_standard', axis=1)


print(f"Original CSV data loaded. Number of rows: {len(df)}")
print(f"Original distribution of labels:")
print(df['label_majority'].value_counts())

# Optimize GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Activate Mixed Precision
if tf.test.is_built_with_cuda():
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed precision policy set to mixed_float16")
else:
    print("CUDA not available, using default precision")




# Pfade und Konfiguration
base_dir =  r"C:\Users\meh91075\Desktop\DataSet(S1)"
filtered_dir = os.path.join(base_dir, 'New_filtered_data_label0')
non_filtered_dir = os.path.join(base_dir, 'New_non_filtered_data')



img_height, img_width = 224, 224
batch_size = 32

def get_image_paths(directory):
    return [os.path.join(root, file) for root, _, files in os.walk(directory) 
            for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

filtered_images = get_image_paths(filtered_dir)
non_filtered_images = get_image_paths(non_filtered_dir)
print(f"Found images: Filtered: {len(filtered_images)}, Not filtered: {len(non_filtered_images)}")

def extract_filename(path):
    basename = os.path.basename(path)
    return basename.replace('_ISO', '').replace('_ISOAlt', '').split('.')[0]

image_df = pd.DataFrame({
    'image_path': filtered_images + non_filtered_images,
    'label': [1] * len(filtered_images) + [0] * len(non_filtered_images),
    'filename': [extract_filename(path) for path in filtered_images + non_filtered_images]
})

print("Sample of extracted filenames:")
print(image_df['filename'].head())

# Adjusted merge process
merged_df = pd.merge(df, image_df[['filename', 'image_path']], on='filename', how='left')
merged_df['label'] = merged_df['label_majority'].map({1: 0, 2: 0, 3: 0, 4: 1})
merged_df['has_image'] = merged_df['image_path'].notna()

# Filter only entries with associated images
merged_df_with_images = merged_df[merged_df['has_image']]

# Undersampling of the majority class
majority_class = merged_df_with_images[merged_df_with_images['label'] == 0]
minority_class = merged_df_with_images[merged_df_with_images['label'] == 1]

majority_downsampled = majority_class.sample(n=2000, random_state=42)
balanced_df = pd.concat([majority_downsampled, minority_class])

print(f"After undersampling: Number of rows: {len(balanced_df)}")
print("Distribution of labels after undersampling:")
print(balanced_df['label'].value_counts())

# Train-Test-Split
train_df, val_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['label'], random_state=42)
print(f"Training data: {len(train_df)}, Validation data: {len(val_df)}")

# Determine the numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_columns = [col for col in numeric_columns if col not in ['label_majority', 'label']]

# Normalization of numeric data
scaler = StandardScaler()
train_df[numeric_columns] = scaler.fit_transform(train_df[numeric_columns])
val_df[numeric_columns] = scaler.transform(val_df[numeric_columns])

#  data augmentation
def create_dataset(dataframe, is_training=True):
    def parse_function(filename, label, *numeric_data):
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_height, img_width])
        img = tf.keras.applications.resnet_v2.preprocess_input(img)
        
        if is_training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
            img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
            img = tf.image.random_hue(img, max_delta=0.1)
        
        numeric_data = tf.convert_to_tensor(numeric_data, dtype=tf.float32)
        numeric_data = tf.squeeze(numeric_data)
        return (img, numeric_data), label

    dataset = tf.data.Dataset.from_tensor_slices((
        dataframe['image_path'].values,
        dataframe['label'].values,
        dataframe[numeric_columns].values
    ))
    
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(dataframe)).repeat()
    else:
        dataset = dataset.repeat()
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Create the datasets
train_dataset = create_dataset(train_df)
val_dataset = create_dataset(val_df, is_training=False)

# Calculate steps per epoch
steps_per_epoch = len(train_df) // batch_size
validation_steps = len(val_df) // batch_size

# Optimized model definition
def create_optimized_hybrid_model(img_height, img_width, num_numeric_features):
    # Image processing branch (CNN)
    img_input = Input(shape=(img_height, img_width, 3))
    
    # Use a pre-trained model for feature extraction (here ResNet50V2)
    base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=img_input)
    
    # Freeze the pre-trained layers
    base_model.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Numerical data branch (MLP)
    num_input = Input(shape=(num_numeric_features,))
    y = layers.Dense(128, activation='relu')(num_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(256, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.4)(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    
    # Combining the branches
    combined = layers.concatenate([x, y])
    
    # Final layers
    z = layers.Dense(256, activation='relu')(combined)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.5)(z)
    z = layers.Dense(128, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.5)(z)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid')(z)
    
    model = models.Model(inputs=[img_input, num_input], outputs=output)
    return model

# Learning Rate Logger Callback
class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            lr = self.model.optimizer.learning_rate
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr = lr(self.model.optimizer.iterations)
            logs['lr'] = tf.keras.backend.get_value(lr)

# Custom  Callback
class CustomReduceLROnPlateau(tf.keras.callbacks.Callback):
    def __init__(self, factor=0.1, patience=10, min_lr=0, monitor='val_loss', mode='min'):
        super(CustomReduceLROnPlateau, self).__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.monitor = monitor
        self.mode = mode
        self.best = np.Inf if mode == 'min' else -np.Inf
        self.wait = 0
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.best = np.Inf if self.mode == 'min' else -np.Inf
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.mode == 'min' and current < self.best:
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        elif self.mode == 'max' and current > self.best:
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            current_lr = self.model.optimizer.learning_rate
            if isinstance(current_lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                current_lr = current_lr(self.model.optimizer.iterations)
            current_lr = float(current_lr.numpy())
            
            if current_lr > self.min_lr:
                new_lr = max(current_lr * self.factor, self.min_lr)
                new_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=new_lr,
                    decay_steps=1000,
                    decay_rate=0.9,
                    staircase=True
                )
                self.model.optimizer.learning_rate = new_lr_schedule
                print(f'\nEpoch {epoch+1}: ReduceLROnPlateau reducing learning rate to {new_lr}.')
                self.wait = 0
                self.model.set_weights(self.best_weights)

# Compile and train
initial_learning_rate = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Multi-GPU Strategy
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_optimized_hybrid_model(img_height, img_width, len(numeric_columns))
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

# Callbacks
reduce_lr = CustomReduceLROnPlateau(monitor='val_auc', factor=0.2, patience=5, min_lr=1e-6, mode='max')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_auc', mode='max')
lr_logger = LearningRateLogger()

# Training
history = model.fit(
    train_dataset,
    epochs=100,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=[reduce_lr, model_checkpoint, lr_logger]
)

# Save model
model.save('hybrid_model_optimized_final_new.h5')

# Evaluation
val_loss, val_accuracy, val_auc, val_precision, val_recall = model.evaluate(val_dataset, steps=validation_steps)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation AUC: {val_auc:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")

# Predictions for  validation dataset
y_pred = model.predict(val_dataset, steps=validation_steps)
y_pred_classes = (y_pred > 0.5).astype("int32")
y_true = np.concatenate([y for _, y in val_dataset.take(validation_steps)], axis=0)

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_true, y_pred_classes)
class_report = classification_report(y_true, y_pred_classes)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

#  training history
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['auc'], label='Training AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# learning rate
plt.figure(figsize=(10, 5))
plt.plot(history.history['lr'])
plt.title('Learning Rate over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.savefig('learning_rate.png')
plt.close()

# confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
plt.close()

print("Visualizations have been saved as PNG files.")

predictions_df = pd.DataFrame({
    'True_Label': y_true,
    'Predicted_Probability': y_pred.flatten(),
    'Predicted_Class': y_pred_classes.flatten()
})
predictions_df.to_csv('predictions.csv', index=False)
print("Predictions have been saved in 'predictions.csv'.")

# Model summary
model.summary()

# Save model architecture as image
tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
print("Model architecture has been saved as 'model_architecture.png'.")

# Calculate and output additional metrics
# ROC AUC
roc_auc = roc_auc_score(y_true, y_pred)
print(f"ROC AUC: {roc_auc:.4f}")

# Precision-Recall AUC
pr_auc = average_precision_score(y_true, y_pred)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# F1-Score 
thresholds = np.arange(0, 1.01, 0.01)
f1_scores = [f1_score(y_true, y_pred > threshold) for threshold in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)
print(f"Best F1-Score: {best_f1:.4f} at threshold {best_threshold:.2f}")

# Visualization Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('precision_recall_curve.png')
plt.close()
print("Precision-Recall curve has been saved as 'precision_recall_curve.png'.")

# Visualization ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.savefig('roc_curve.png')
plt.close()
print("ROC curve has been saved as 'roc_curve.png'.")

# Save training history
with open('training_history.txt', 'w') as f:
    for key, values in history.history.items():
        f.write(f"{key}: {values}\n")
print("Training history has been saved in 'training_history.txt'.")

# Analyze misclassified examples
misclassified = np.where(y_pred_classes.flatten() != y_true)[0]
print(f"\nNumber of misclassified examples: {len(misclassified)}")

if len(misclassified) > 0:
    print("\nDetails of some misclassified examples:")
    for i in misclassified[:5]:  # Show details for the first 5 misclassified examples
        true_label = y_true[i]
        pred_label = y_pred_classes[i][0]
        pred_prob = y_pred[i][0]
        print(f"Example {i}: True Label: {true_label}, Predicted Label: {pred_label}, Predicted Probability: {pred_prob:.4f}")

# Model performance across different thresholds
accuracies = [accuracy_score(y_true, y_pred > threshold) for threshold in thresholds]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracies, label='Accuracy')
plt.plot(thresholds, f1_scores, label='F1-Score')
plt.axvline(x=0.5, color='r', linestyle='--', label='Default Threshold (0.5)')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Model Performance vs. Classification Threshold')
plt.legend()
plt.savefig('performance_vs_threshold.png')
plt.close()
print("Performance vs. Threshold plot has been saved as 'performance_vs_threshold.png'.")


try:
    import shap
    
    # Select a subset of validation data for SHAP analysis
    num_explain = 100
    explain_dataset = val_dataset.take(num_explain).unbatch()
    background_dataset = train_dataset.take(num_explain).unbatch()
    
    # Create an Explainer
    explainer = shap.DeepExplainer(model, background_dataset)
    
    # Calculate SHAP 
    shap_values = explainer.shap_values(explain_dataset)
    
    # Visualize (SHAP )
    shap.summary_plot(shap_values[0], plot_type="bar", feature_names=numeric_columns, show=False)
    plt.savefig('shap_summary.png')
    plt.close()
    print("SHAP summary has been saved as 'shap_summary.png'.")
except ImportError:
    print("SHAP is not installed. Skipping SHAP analysis.")

print("\nAll analyses and visualizations are complete.")

# Summary (metrics)
print("\nSummary of key metrics:")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation AUC: {val_auc:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")
print(f"Best F1-Score: {best_f1:.4f} at threshold {best_threshold:.2f}")

# Save summary (text-file)
with open('model_summary.txt', 'w') as f:
    f.write("Summary of key metrics:\n")
    f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
    f.write(f"Validation AUC: {val_auc:.4f}\n")
    f.write(f"Validation Precision: {val_precision:.4f}\n")
    f.write(f"Validation Recall: {val_recall:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
    f.write(f"Precision-Recall AUC: {pr_auc:.4f}\n")
    f.write(f"Best F1-Score: {best_f1:.4f} at threshold {best_threshold:.2f}\n")

print("\nA summary of key metrics has been saved in 'model_summary.txt'.")
print("\nThe script has been executed successfully. All results and visualizations have been saved.")

