import os
import sys
import customtkinter as ctk
import math
from PIL import Image
import json
import pandas as pd


# ---------- CLASSES ----------
class Gui:
    def __init__(self, master, row, column, padx=5, pady=5):
        self.subframe, self.optionmenu, self.button, self.entry = {}, {}, {}, {}
        self.selected_radiobutton = None
        self.frame = ctk.CTkFrame(master=master)
        self.frame.grid(row=row, column=column, padx=padx, pady=pady)

    def create_subframe(self, name, row, column, padx=5, pady=5):
        self.subframe[name] = Gui(master=self.frame, row=row, column=column, padx=padx, pady=pady)
        return self

    def create_text(self, text, row, column, columnspan=1, padx=1, pady=1):
        text = ctk.CTkLabel(master=self.frame, text=text, font=('', 15,))
        text.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady)
        return self

    def create_header(self, text, row, column, columnspan=1, padx=5, pady=5):
        header = ctk.CTkLabel(master=self.frame, text=text, font=('', 18, 'bold'))
        header.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady)
        return self

    def create_entry(self, name, row, column, columnspan=1, placeholder_text=None, state='normal'):
        self.entry[name] = ctk.CTkEntry(master=self.frame, placeholder_text=placeholder_text, state=state)
        self.entry[name].grid(row=row, column=column, columnspan=columnspan)
        return self

    def create_button(self, name, text, command, row, column, state='normal', columnspan=1):
        self.button[name] = ctk.CTkButton(master=self.frame, text=text,
                                          command=command, state=state)
        self.button[name].grid(row=row, column=column, columnspan=columnspan, padx=5, pady=5)
        return self


# ---------- PART CLASS ----------
class LabelingPart:
    # INIT:
    def __init__(self, index):
        # Paths:
        self.index = index
        self.info_path = os.path.join(info_path, info_files[self.index])
        self.part_name = os.path.basename(self.info_path)[:-5]
        self.image_ISO_path = os.path.join(png_path, f'{self.part_name}_ISO.png')
        self.image_ISOAlt_path = os.path.join(png_path, f'{self.part_name}_ISOAlt.png')
        # self.image_TOP_path = os.path.join(png_path, f'{self.part_name}_TOP.png')
        # self.image_FRONT_path = os.path.join(png_path, f'{self.part_name}_FRONT.png')
        # self.image_RIGHT_path = os.path.join(png_path, f'{self.part_name}_RIGHT.png')
        if len(step_files) > self.index:
            self.step_path = os.path.join(step_path, step_files[self.index])
        else:
            self.step_path = None

        # Labels:
        if index < len(df):
            label_data = df.iloc[self.index]['label']
            label_map = {0: 'UNDEFINED PART', 1: 'AM PART', 2: 'MILLING PART', 3: 'SHEET METAL PART', 4: 'STANDARD COMPONENT'}
            if label_data in label_map:
                self.label = f'{label_data} - {label_map[label_data]}'
            else:
                self.label = 'ERROR'
        else:
            self.label = None

        # Part info:
        with open(self.info_path, 'r') as f:
            part_info = json.load(f)
            self.part_xdim = math.ceil(float(part_info['xSize']))
            self.part_ydim = math.ceil(float(part_info['ySize']))
            self.part_zdim = math.ceil(float(part_info['zSize']))

        # Progress:
        self.progress_value = self.index / len(info_files)

    # DISPLAY:
    def display_picture(self):
        part_image_ISO = Image.open(self.image_ISO_path)
        part_image_ISOAlt = Image.open(self.image_ISOAlt_path)
        # part_image_TOP = Image.open(self.image_TOP_path)
        # part_image_RIGHT = Image.open(self.image_RIGHT_path)
        # part_image_FRONT = Image.open(self.image_FRONT_path)
        part_image_ISO = part_image_ISO.resize((width_ISO, height_ISO))
        part_image_ISOAlt = part_image_ISOAlt.resize((width_ISO, height_ISO))
        # part_image_TOP = part_image_TOP.resize((width_VIEWS, height_VIEWS))
        # part_image_RIGHT = part_image_RIGHT.resize((width_VIEWS, height_VIEWS))
        # part_image_FRONT = part_image_FRONT.resize((width_VIEWS, height_VIEWS))
        # Display image:
        picture_ISO_ctk.configure(light_image=part_image_ISO)
        picture_ISOAlt_ctk.configure(light_image=part_image_ISOAlt)
        # picture_TOP_ctk.configure(light_image=part_image_TOP)
        # picture_RIGHT_ctk.configure(light_image=part_image_RIGHT)
        # picture_FRONT_ctk.configure(light_image=part_image_FRONT)
        picture_ISO.configure(image=picture_ISO_ctk)
        picture_ISOAlt.configure(image=picture_ISOAlt_ctk)
        # picture_TOP.configure(image=picture_TOP_ctk)
        # picture_RIGHT.configure(image=picture_RIGHT_ctk)
        # picture_FRONT.configure(image=picture_FRONT_ctk)

    def display_labels(self):
        # Set labels:
        part_labels = [self.part_name, self.label, self.index]
        # Display labels:
        for i, part_value in enumerate(values_label):
            part_value.configure(text=str(part_labels[i]))

    def display_info(self):
        # Display info data:
        dim_label.configure(text=f'{self.part_xdim}mm x {self.part_ydim}mm x {self.part_zdim}mm')

    def display_progress(self):
        # Update progress accordingly:
        progressbar.set(self.progress_value)
        progressbar_header.configure(text=f'PROGRESS: {round(self.progress_value * 100, 2)}%')
        # Check for last part:
        if self.index == len(png_files) - 1:
            if len(df) < len(png_files):
                progressbar.set(self.progress_value)
            elif len(df) == len(png_files):
                progressbar.set(1)
                progressbar_header.configure(text='PROGRESS: 100%')

    # METHODS:
    def label_part(self, label_num):
        global df, file_index, last_file_index
        if 0 <= self.index < len(png_files):
            if self.label is None:
                # Update part label:
                self.label = label_num
                # Create new row for df:
                data = {'ID': self.index, 'filename': [self.part_name], 'label': [self.label]}
                df_new_label = pd.DataFrame(data)
                # Append the new row to the existing CSV file
                df_new_label.to_csv(label_path, mode='a', header=False, index=False)
                # Update existing df and turn ID and label into int:
                df = pd.concat([df, df_new_label]).astype({'ID': int, 'label': int})
                # Print message
                label_map = {
                    1: 'AM part',
                    2: 'milling part',
                    3: 'sheet metal part',
                    4: 'standard component',
                    0: 'undefined part'}
                if label_num in label_map:
                    print(f'-> Part labeled as {label_map[label_num]}.')
                    # Load next part automatically:
                    if file_index < len(png_files) - 1:
                        file_index += 1
                        if last_file_index < file_index:
                            last_file_index = file_index
                        load_part(file_index)
                    # Notify when last part was labeled:
                    else:
                        console_print('LABELING COMPLETED! All parts are labeled.')
                        # Update Display:
                        progressbar.set(value=1)
                        progressbar_header.configure(text='PROGRESS: 100%')
                        self.display_labels()
            elif self.label is not None and last_file_index != len(png_files):
                print('-> This part is already labeled.')
            else:
                error_print('No part currently loaded.')
        else:
            error_print('Part index out of range.')

    def change_label(self):
        global df
        if self.label is not None:
            # Open Input box:
            dialog = ctk.CTkInputDialog(text="Choose new label from: 1, 2, 3, 4, 0", title="Change Label")
            new_label_num = int(dialog.get_input())  # waits for input
            if new_label_num in {0, 1, 2, 3, 4}:
                # Assign new label to df and part:
                df.loc[df['ID'] == self.index, 'label'] = new_label_num
                self.label = new_label_num
                # Update displayed label:
                self.display_labels()
                # Write the updated df to the existing CSV file
                df.to_csv(label_path, index=False)
                # Print message
                label_map = {
                    1: 'AM part',
                    2: 'milling part',
                    3: 'sheet metal part',
                    4: 'standard component',
                    0: 'undefined part'}
                if new_label_num in label_map:
                    print(f'-> Part labeled as {label_map[new_label_num]}.')
            else:
                error_print('Chosen label is not defined!',
                            helptext='Choose label from: 0, 1, 2, 3, 4')
        else:
            error_print('Current part has no label to be changed.',
                        helptext='Please label with the label buttons.')

    def open_cad(self):
        print('#### Opening 3D Model is not implemented for this version! ####')
        pass
        # if self.step_path is not None:
        #     # Initialize display:
        #     display, start_display, add_menu, add_function_to_menu = init_display(
        #         None, (1024, 768), True, [206, 215, 222], [180, 180, 180])
        #
        #     # Read step file:
        #     shp = read_step_file(self.step_path)
        #     t = TopologyExplorer(shp)
        #     face = []
        #     for f in t.faces():
        #         face.append(f)
        #
        #     # Display part:
        #     display.DisplayShape(face, update=True)
        #     start_display()
        # else:
        #     print('No STEP file for this model available. '
        #           'Please place the corresponding file into the step folder if you want to view it.')


# ---------- FUNCS ----------
def console_print(text):
    """ This function prints formatted content to the console."""
    print("\n" + 100*"-")
    print(f"{text}".center(100))
    print(100*"-")


def error_print(error, helptext=''):
    """ This function prints a formatted error message to the console."""
    len_text = 20 + max(len(error), len(helptext))
    print("\n" + (len_text+4)*"#")
    print('#', f'ERROR: {error}'.center(len_text), '#')
    if helptext != '':
        print('#', helptext.center(len_text), '#')
    print((len_text+4)*"#")


def get_user_id():
    return main_gui.subframe['user_id'].entry['user_id'].get().upper()


def submit_id():
    global user_id
    # Gather USER ID from entry:
    user_id = get_user_id()

    try:
        # Check USER ID criteria:
        if len(user_id) != 2 or not user_id.isalpha():
            raise ValueError('Your USER ID has to be 2 characters in length and be only of alphabetical.')
        print(f'\nUser ID "{user_id}" accepted.\nContinue with application selection.')
        # Unlock labeling apps for all user:
        for button_i in main_gui.subframe['labeling_apps'].button:
            main_gui.subframe['labeling_apps'].button[button_i].configure(state='normal')

    except FileNotFoundError as e:
        error_print(str(e))


# ---------- FUNCTIONS ----------
def load_part(index):
    global current_part
    PartClass = LabelingPart
    if index < len(png_files):
        # Load part:
        current_part = PartClass(index)
        current_part.display_picture()
        current_part.display_labels()
        current_part.display_info()
        current_part.display_progress()
        # Print to console:
        print('\n', 10*'#', f'\nID: {index}, FILE: {current_part.part_name}')


# ---------- BUTTON FUNCTIONS ----------
def button_new_labeling():
    global continue_flag, user_id
    label_path = os.path.join(csv_path, 'labels_' + str(get_user_id()) + '.csv')

    # Make sure existing label_userID.csv is not accidentally overwritten:
    if os.path.isfile(label_path):
        # Open Input box:
        dialog = ctk.CTkInputDialog(text='Warning: You are about to override your existing labels!'
                                         '\n\nALL YOUR EXISTING LABELS WILL BE DELETED! '
                                         'Consider a manual backup before proceeding.'
                                         "\n\nType in 'restart labeling' to continue.",
                                    title="Overriding Current Labels")
        dialog_answer = str(dialog.get_input())  # waits for input

        # IF 'restart labeling':
        if dialog_answer == 'restart labeling':
            continue_flag = False
            app.destroy()
        else:
            print('Restart of labeling was aborted!')
    # IF labels not already exits simply start labeling:
    else:
        console_print('Starting autosplit_labeling.py ...')
        continue_flag = False
        app.destroy()



def button_continue_labeling():
    global continue_flag, user_id
    label_path = os.path.join(csv_path, 'labels_' + str(get_user_id()) + '.csv')

    # Make sure existing label_userID.csv is not accidentally overwritten:
    if os.path.isfile(label_path):
        # Start autosplit_labeling.py from existing label.csv:
        console_print('Starting autosplit_labeling.py ...')
        continue_flag = True
        app.destroy()
    else:
        console_print(f'Cant continue, no csv file for User ID {user_id} was found.')


def button_assignlabel(label_num):
    return lambda: current_part.label_part(label_num)


def button_previous_file():
    global file_index
    if file_index > 0:
        file_index -= 1
        # Load part data:
        load_part(file_index)
    else:
        error_print('No previous file existing.')


def button_next_file():
    global file_index, last_file_index
    if file_index < len(png_files) - 1:
        # Make it unable to skip part before labeling:
        if file_index < last_file_index:
            file_index += 1
            # Load part data:
            load_part(file_index)
        else:
            print("Please label the current file before proceeding!")
    else:
        print('Last part loaded.')


def button_open_cad():
    global current_part
    current_part.open_cad()


def button_change_label():
    current_part.change_label()


def hotkeys(event):
    hotkey_map = {
        'n': buttons_hk['(N)'],
        'b': buttons_hk['(B)'],
        '1': buttons_hk['(1)'],
        '2': buttons_hk['(2)'],
        '3': buttons_hk['(3)'],
        '4': buttons_hk['(4)'],
        '0': buttons_hk['(0)']
    }
    if event.char in hotkey_map:
        hotkey_map[event.char].invoke()


if __name__ == '__main__':
    # ---------- ADMINISTRATORS ----------
    admin_list = ['RN', 'FS']
    user_id = None
    continue_flag = None

    # ---------- DATA PATHS ----------
    root_path = os.path.dirname(os.path.abspath(__file__))
    png_path, info_path, step_path, csv_path = [os.path.join(root_path, 'data', folder)
                                                for folder in ('png', 'info', 'step', 'csv')]
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)

    # ---------- GATHER USER ID ----------
    ctk.set_appearance_mode('light')
    ctk.set_default_color_theme('green')
    app = ctk.CTk()
    app.title('Autosplit')
    # Main menu:
    main_gui = Gui(master=app, row=0, column=0)
    main_gui.create_header('AutoSplit Labeling', 0, 0)
    # User ID Section:
    main_gui.create_subframe('user_id', 1, 0)
    main_gui.subframe['user_id'].create_text('USER ID:', 0, 0).create_entry('user_id', 0, 1, placeholder_text='YOUR USER ID').create_button('submit_id', 'Submit ID', command=submit_id, row=0, column=2)
    # Labeling Apps Section:
    main_gui.create_subframe('labeling_apps', 2, 0)
    main_gui.subframe['labeling_apps'].create_header('LABELING APPS', 0, 0, columnspan=2).create_button('new_labeling', 'START NEW LABEL FILE', command=button_new_labeling, row=1, column=0, state='disabled').create_button(
        'continue_labeling', 'CONTINUE LABELING', command=button_continue_labeling, row=1, column=1, state='disabled')
    app.mainloop()

    # ---------- LABELING ----------
    png_files, info_files, step_files = [os.listdir(folder_path) for folder_path in (png_path, info_path, step_path)]

    # Output definition:
    df_labels = {'ID': [], 'filename': [], 'label': []}

    # File index initialization:
    file_index, last_file_index = 0, 0

    # ---------- DETERMINE WHETHER TO START, CONTINUE OR REVIEW LABELING ----------
    # IF standard mode:
    console_print('AUTOSPLIT LABELING APPLICATION')
    label_path = os.path.join(csv_path, 'labels_' + user_id + '.csv')
    # IF labeling is to be continued:
    if continue_flag and os.path.exists(label_path):
        df = pd.read_csv(label_path)
        # IF .csv isn't empty, set file_index to ID of last row to continue labeling:
        if len(df) < len(png_files) and not len(df) == 0:
            file_index = df.iloc[-1]['ID'] + 1
            last_file_index = file_index
            print(10 * '#', f'\nYou have already labeled some parts.')
            print(f'Labeling will continue from ID: {file_index}, part name: {png_files[file_index][:-4]}.')
        # IF df in .csv is empty, simply start from index = 0:
        if len(df) == 0:
            print(10 * '#', '\nYour existing label_userID.csv is empty.')
            print('Labeling will commence from first part')
        # IF all parts are already labeled:
        elif len(df) == len(png_files):
            console_print('LABELING COMPLETED! All parts are already labeled.')
            file_index = df.iloc[-1]['ID']
            last_file_index = file_index
    # IF labeling is to start from scratch:
    elif not continue_flag:
        df = pd.DataFrame(df_labels)
        df.to_csv(label_path, index=False)
    # Error handling:
    elif continue_flag and not os.path.exists(label_path):
        error_print("No existing label_userID.csv file found!",
                    helptext="Either check for mistype in userID or you haven't began labeling yet.")
        sys.exit()

    # Initialize first part according to labeling mode:
    current_part = LabelingPart(index=file_index)
    print(10 * '#', f'\nID: {file_index}, FILE: {current_part.part_name}')

    # ---------- MAIN GUI ----------
    # Root frame:
    root = ctk.CTk()
    # Main title:
    title = ctk.CTkLabel(master=root,
                         text='AutoSplit - Part Categorizer', font=('', 20, 'bold'))
    title.grid(row=0, column=0, columnspan=2)
    # Create sub frames:
    frame_names = [
        # Format: (name, row, column, width, height)
        ('image_frame', 1, 0, 500, 500),
        ('labeldata_frame', 1, 1, 200, 200),
        ('dim_frame', 2, 0, 1000, 200),
        ('progressbar_frame', 2, 1, 200, 200),
        ('labelbutton_frame', 3, 0, 200, 200),
        ('controlbutton_frame', 3, 1, 200, 200)]
    frames_ctk = {}
    for (name, row, column, width, height) in frame_names:
        frames_ctk[name] = ctk.CTkFrame(master=root, width=width, height=height)
        frames_ctk[name].grid(row=row, column=column, padx=10, pady=10)

    # ---------- IMAGE ----------
    # Create sub frame for each view:
    picture_frame_ISO = ctk.CTkFrame(master=frames_ctk['image_frame'], width=100, height=100)
    picture_frame_ISOAlt = ctk.CTkFrame(master=frames_ctk['image_frame'], width=100, height=100)
    # picture_frame_TOP = ctk.CTkFrame(master=frames_ctk['image_frame'], width=50, height=50)
    # picture_frame_RIGHT = ctk.CTkFrame(master=frames_ctk['image_frame'], width=50, height=50)
    # picture_frame_FRONT = ctk.CTkFrame(master=frames_ctk['image_frame'], width=100, height=100)
    picture_frame_ISO.grid(row=0, column=0, padx=5, pady=5)
    picture_frame_ISOAlt.grid(row=0, column=1, padx=5, pady=5)
    # picture_frame_TOP.grid(row=1, column=0, padx=5, pady=5)
    # picture_frame_RIGHT.grid(row=1, column=1, padx=5, pady=5)
    # picture_frame_FRONT.grid(row=1, column=2, padx=5, pady=5)
    image_ISO = Image.open(current_part.image_ISO_path)
    image_ISOAlt = Image.open(current_part.image_ISOAlt_path)
    # image_TOP = Image.open(current_part.image_TOP_path)
    # image_RIGHT = Image.open(current_part.image_RIGHT_path)
    # image_FRONT = Image.open(current_part.image_FRONT_path)
    # Resize image:
    scale_percent_ISO, scale_percent_VIEWS = 50, 15
    width_ISO, height_ISO = int(image_ISO.width * scale_percent_ISO / 100), int(image_ISO.height * scale_percent_ISO / 100)
    # width_VIEWS, height_VIEWS = int(image_ISO.width * scale_percent_VIEWS / 100), int(image_ISO.height * scale_percent_VIEWS / 100)
    image_ISO = image_ISO.resize((width_ISO, height_ISO))
    image_ISOAlt = image_ISOAlt.resize((width_ISO, height_ISO))
    # image_TOP = image_TOP.resize((width_VIEWS, height_VIEWS))
    # image_RIGHT = image_RIGHT.resize((width_VIEWS, height_VIEWS))
    # image_FRONT = image_FRONT.resize((width_VIEWS, height_VIEWS))
    # Load image:
    picture_ISO_ctk = ctk.CTkImage(light_image=image_ISO, size=(width_ISO, height_ISO))
    picture_ISOAlt_ctk = ctk.CTkImage(light_image=image_ISOAlt, size=(width_ISO, height_ISO))
    # picture_TOP_ctk = ctk.CTkImage(light_image=image_TOP, size=(width_VIEWS, height_VIEWS))
    # picture_RIGHT_ctk = ctk.CTkImage(light_image=image_RIGHT, size=(width_VIEWS, height_VIEWS))
    # picture_FRONT_ctk = ctk.CTkImage(light_image=image_FRONT, size=(width_VIEWS, height_VIEWS))
    picture_ISO = ctk.CTkLabel(master=picture_frame_ISO, image=picture_ISO_ctk, text='')
    picture_ISOAlt = ctk.CTkLabel(master=picture_frame_ISOAlt, image=picture_ISOAlt_ctk, text='')
    # picture_TOP = ctk.CTkLabel(master=picture_frame_TOP, image=picture_TOP_ctk, text='')
    # picture_RIGHT = ctk.CTkLabel(master=picture_frame_RIGHT, image=picture_RIGHT_ctk, text='')
    # picture_FRONT = ctk.CTkLabel(master=picture_frame_FRONT, image=picture_FRONT_ctk, text='')
    picture_ISO.grid(row=0, column=0)
    picture_ISOAlt.grid(row=0, column=0)
    # picture_TOP.grid(row=0, column=0)
    # picture_RIGHT.grid(row=0, column=0)
    # picture_FRONT.grid(row=0, column=0)

    # ---------- LABEL DATA DISPLAY ----------
    labeldata_header = ctk.CTkLabel(master=frames_ctk['labeldata_frame'],
                                    text='CURRENT PART LABEL:', font=('', 16, 'bold'))
    labeldata_header.grid(row=0, column=0, columnspan=2, pady=10)
    # Labels 'NAME', 'LABEL', 'INDEX':
    labels = [('Name:', current_part.part_name), ('Label:', current_part.label), ('Index:', current_part.index)]
    values_label = []
    for row, (heading, value) in enumerate(labels, start=1):
        heading_label = ctk.CTkLabel(master=frames_ctk['labeldata_frame'], text=heading, font=('', 14))
        heading_label.grid(row=row, column=0, padx=10, sticky='w')
        value_label = ctk.CTkLabel(master=frames_ctk['labeldata_frame'], text=str(value), font=('', 14))
        values_label.append(value_label)
        value_label.grid(row=row, column=1, padx=10, sticky='w')

    # ---------- INFO DATA DISPLAY ----------
    dim_label_header = ctk.CTkLabel(master=frames_ctk['dim_frame'],
                                    text='Dimensions:', font=('', 14))
    dim_label_header.pack(side='left', padx=10)
    dim_label = ctk.CTkLabel(master=frames_ctk['dim_frame'],
                             text=f'{current_part.part_xdim}mm x {current_part.part_ydim}mm x {current_part.part_zdim}mm',
                             font=('', 14, 'bold'))
    dim_label.pack(side='right', padx=10)

    # ---------- PROGRESSBAR ----------
    progressbar_header = ctk.CTkLabel(master=frames_ctk['progressbar_frame'],
                                      text=f'PROGRESS: {round(current_part.progress_value * 100, 2)}%',
                                      font=('', 16, 'bold'))
    progressbar_header.grid(row=0, column=0)
    progressbar = ctk.CTkProgressBar(master=frames_ctk['progressbar_frame'], orientation="horizontal", height=20,
                                     width=300)
    progressbar.grid(row=1, column=0)
    # Set initial progress accordingly:
    if len(df) < len(png_files):
        progressbar.set(current_part.progress_value)
    elif len(df) == len(png_files):
        progressbar.set(1)
        progressbar_header.configure(text='PROGRESS: 100%')

    # ---------- LABEL BUTTONS ----------
    # Heading "LABELS:"
    header_labelbutton = ctk.CTkLabel(master=frames_ctk['labelbutton_frame'],
                                      text='LABELS:', font=('', 16, 'bold'))
    header_labelbutton.grid(row=0, column=0, columnspan=2)
    # Label button data:
    labelbutton_data = [
        # Format: ("button_name (Hotkey)", command(label_num), row, column)
        ("AM PART (1)", button_assignlabel(1), 1, 0),
        ("MILLING PART (2)", button_assignlabel(2), 1, 1),
        ("SHEET METAL PART (3)", button_assignlabel(3), 2, 0),
        ("STANDARD COMPONENT (4)", button_assignlabel(4), 2, 1),
        ("UNDEFINED PART (0)", button_assignlabel(0), 3, 0)]
    # Place and assign label buttons:
    buttons_hk = {}
    for (button_name, command, row, column) in labelbutton_data:
        labelbutton = ctk.CTkButton(master=frames_ctk['labelbutton_frame'], text=button_name,
                                    command=command, height=30, width=200)
        labelbutton.grid(row=row, column=column, padx=2, pady=2)
        # Read labeling hotkeys:
        buttons_hk[button_name.split(' ')[-1]] = labelbutton

    # ---------- CONTROL BUTTONS ----------
    # Heading "CONTROL:"
    header_controlbutton = ctk.CTkLabel(master=frames_ctk['controlbutton_frame'],
                                        text='CONTROLS:', font=('', 16, 'bold'))
    header_controlbutton.grid(row=0, column=0, columnspan=2)
    # Control button data:
    controlbutton_data = [
        # Format: ("button_name (Hotkey)", command(label_num), row, column)
        ("PREVIOUS PART (B)", button_previous_file, 1, 0),
        ("NEXT PART (N)", button_next_file, 1, 1),
        ("OPEN 3D MODEL", button_open_cad, 2, 0),
        ("CHANGE LABEL", button_change_label, 2, 1)]
    # Place and assign button:
    for (button_name, command, row, column) in controlbutton_data:
        controlbutton = ctk.CTkButton(master=frames_ctk['controlbutton_frame'], text=button_name,
                                      command=command, height=30, width=200)
        controlbutton.grid(row=row, column=column, padx=2, pady=2)
        # Read control hotkeys:
        buttons_hk[button_name.split(' ')[-1]] = controlbutton

    # Bind hotkeys
    root.bind("<Key>", hotkeys)

    # Start main loop:
    root.mainloop()