import xml.etree.ElementTree as ET
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
import os
import cv2  # Import OpenCV
import numpy as np
import traceback  # For detailed error printing

RELATIONSHIP_LABELS_FILE = "relationship_labels.json"
UNDO_REDO_MAX_SIZE = 20
# Define colors for masks and bounding boxes (BGR format for OpenCV)
SOURCE_MASK_COLOR = (0, 0, 255)  # Red
SOURCE_BBOX_COLOR = (0, 255, 0)  # Green
TARGET_MASK_COLOR = (255, 0, 0)  # Blue
TARGET_BBOX_COLOR = (0, 255, 255)  # Yellow

# --- Dark Theme Colors ---
DARK_MODE_BG = "#2e2e2e"
DARK_MODE_FRAME_BG = "#3c3c3c"  # Slightly lighter for frames/containers
DARK_MODE_WIDGET_BG = "#4a4a4a"  # For entry fields, listboxes
DARK_MODE_TEXT_FG = "#d0d0d0"  # Light gray for text
DARK_MODE_SELECT_BG = "#0078d4"  # A distinct selection blue
DARK_MODE_SELECT_FG = "#ffffff"  # White text on selection
DARK_MODE_BUTTON_BG = "#505050"
DARK_MODE_BUTTON_ACTIVE_BG = "#606060"
DARK_MODE_DISABLED_FG = "#777777"
DARK_MODE_BORDER_COLOR = "#555555"  # For frame borders
DARK_MODE_INSERT_COLOR = "#e0e0e0"  # Cursor color


class SceneGraphAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Scene Graph Annotation Tool")

        # Apply Dark Theme
        self.apply_dark_theme()  # Call before creating widgets

        # Data Structures
        self.image_data = {}
        self.relationship_labels = set()
        self.current_image = None
        self.selected_mask_id = None
        self.selected_target_mask_id = None
        self.loaded_xml_path = None
        self.undo_stack = []
        self.redo_stack = []
        self.image_directory = None
        self.tk_image = None
        self._target_mask_map = {}
        self._resize_timer = None
        self.autocomplete_listbox = None

        # GUI Elements
        self.create_widgets()
        self.load_relationship_labels()

    def apply_dark_theme(self):
        """Applies a dark theme to the application."""
        self.root.configure(bg=DARK_MODE_BG)
        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')  # 'clam' is often good for custom theming

        # General widget styling
        self.style.configure('.',
                             background=DARK_MODE_BG,
                             foreground=DARK_MODE_TEXT_FG,
                             fieldbackground=DARK_MODE_WIDGET_BG,  # Default for entry-like fields
                             borderwidth=1)
        self.style.map('.',
                       background=[('disabled', DARK_MODE_BG), ('active', DARK_MODE_FRAME_BG)],
                       foreground=[('disabled', DARK_MODE_DISABLED_FG)])

        # Frame styling
        self.style.configure('TFrame', background=DARK_MODE_FRAME_BG)
        # Specific frame for image to have a slightly different or same bg
        self.style.configure('Image.TFrame', background=DARK_MODE_BG, relief='solid', borderwidth=1,
                             bordercolor=DARK_MODE_BORDER_COLOR)
        self.style.configure('Grooved.TFrame', background=DARK_MODE_FRAME_BG, relief='solid', borderwidth=1,
                             bordercolor=DARK_MODE_BORDER_COLOR)

        # Button styling
        self.style.configure('TButton',
                             background=DARK_MODE_BUTTON_BG,
                             foreground=DARK_MODE_TEXT_FG,
                             relief='raised',
                             borderwidth=1,
                             focuscolor=DARK_MODE_TEXT_FG,  # Color of focus highlight
                             lightcolor=DARK_MODE_BUTTON_BG,  # Used for 3D effect, match bg
                             darkcolor=DARK_MODE_BUTTON_BG)  # Used for 3D effect, match bg
        self.style.map('TButton',
                       background=[('active', DARK_MODE_BUTTON_ACTIVE_BG), ('disabled', DARK_MODE_FRAME_BG)],
                       foreground=[('disabled', DARK_MODE_DISABLED_FG)])

        # Label styling
        self.style.configure('TLabel', background=DARK_MODE_FRAME_BG, foreground=DARK_MODE_TEXT_FG)  # Labels on frames
        self.style.configure('Status.TLabel', background=DARK_MODE_BG, foreground=DARK_MODE_TEXT_FG)  # Status bar label
        self.style.configure('Input.TLabel', background=DARK_MODE_FRAME_BG,
                             foreground=DARK_MODE_TEXT_FG)  # Labels in input_frame
        self.style.configure('ImageCounter.TLabel', background=DARK_MODE_BG, foreground=DARK_MODE_TEXT_FG,
                             font=('TkDefaultFont', 10, 'bold'))  # Style for the new image counter
        self.style.configure('Stats.TLabel', background=DARK_MODE_FRAME_BG, foreground=DARK_MODE_TEXT_FG)

        # Entry styling
        self.style.configure('TEntry',
                             fieldbackground=DARK_MODE_WIDGET_BG,
                             foreground=DARK_MODE_TEXT_FG,
                             insertcolor=DARK_MODE_INSERT_COLOR,  # Cursor color
                             borderwidth=1,
                             relief='solid')  # Use solid relief for better dark theme look
        self.style.map('TEntry',
                       foreground=[('disabled', DARK_MODE_DISABLED_FG)],
                       fieldbackground=[('disabled', DARK_MODE_FRAME_BG)])

        # Combobox styling
        self.style.configure('TCombobox',
                             fieldbackground=DARK_MODE_WIDGET_BG,
                             foreground=DARK_MODE_TEXT_FG,
                             selectbackground=DARK_MODE_WIDGET_BG,  # Background of the dropdown list items
                             selectforeground=DARK_MODE_TEXT_FG,
                             arrowcolor=DARK_MODE_TEXT_FG,
                             insertcolor=DARK_MODE_INSERT_COLOR,
                             relief='solid',
                             borderwidth=1)
        self.style.map('TCombobox',
                       fieldbackground=[('readonly', DARK_MODE_WIDGET_BG), ('disabled', DARK_MODE_FRAME_BG)],
                       foreground=[('disabled', DARK_MODE_DISABLED_FG)],
                       selectbackground=[('readonly', DARK_MODE_WIDGET_BG)],
                       selectforeground=[('readonly', DARK_MODE_TEXT_FG)])

        # For the Combobox dropdown list itself (which is a Toplevel Listbox)
        # This needs to be done using option_add for the popdown listbox
        self.root.option_add('*TCombobox*Listbox.background', DARK_MODE_WIDGET_BG)
        self.root.option_add('*TCombobox*Listbox.foreground', DARK_MODE_TEXT_FG)
        self.root.option_add('*TCombobox*Listbox.selectBackground', DARK_MODE_SELECT_BG)
        self.root.option_add('*TCombobox*Listbox.selectForeground', DARK_MODE_SELECT_FG)

        # Menu styling (can be tricky, sometimes needs global option_add)
        self.root.option_add('*Menu.background', DARK_MODE_FRAME_BG)
        self.root.option_add('*Menu.foreground', DARK_MODE_TEXT_FG)
        self.root.option_add('*Menu.activeBackground', DARK_MODE_SELECT_BG)
        self.root.option_add('*Menu.activeForeground', DARK_MODE_SELECT_FG)
        self.root.option_add('*Menu.disabledForeground', DARK_MODE_DISABLED_FG)
        self.root.option_add('*Menu.relief', 'flat')
        self.root.option_add('*Menu.borderwidth', 0)

        self.root.option_add('*Menubutton.background', DARK_MODE_FRAME_BG)
        self.root.option_add('*Menubutton.foreground', DARK_MODE_TEXT_FG)
        self.root.option_add('*Menubutton.activeBackground', DARK_MODE_SELECT_BG)
        self.root.option_add('*Menubutton.activeForeground', DARK_MODE_SELECT_FG)

    def create_widgets(self):
        """Creates and arranges all the GUI widgets."""
        self.main_frame = ttk.Frame(self.root, padding="10", style='TFrame')
        self.main_frame.grid(row=0, column=0, sticky=(N, W, E, S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Image Frame (Left)
        self.image_frame = ttk.Frame(self.main_frame, style='Image.TFrame')
        self.image_frame.grid(row=0, column=0, rowspan=5, sticky=(N, W, E, S))
        self.main_frame.columnconfigure(0, weight=3)
        self.main_frame.rowconfigure(0, weight=1)

        # --- NEW: Image Counter Label ---
        self.image_counter_label = ttk.Label(self.image_frame, text="Image 0/0", style='ImageCounter.TLabel')
        self.image_counter_label.pack(side=TOP, fill=X, padx=5, pady=(5, 2))

        # Right Column Frames
        self.list_frame = ttk.Frame(self.main_frame, style='Grooved.TFrame', padding=5)
        self.list_frame.grid(row=0, column=1, sticky=(N, W, E, S), padx=(5, 0))
        self.detail_frame = ttk.Frame(self.main_frame, style='Grooved.TFrame', padding=5)
        self.detail_frame.grid(row=1, column=1, sticky=(N, W, E, S), padx=(5, 0), pady=5)
        self.input_frame = ttk.Frame(self.main_frame, style='Grooved.TFrame', padding=5)
        self.input_frame.grid(row=2, column=1, sticky=(N, W, E, S), padx=(5, 0), pady=5)
        self.button_frame = ttk.Frame(self.main_frame, style='Grooved.TFrame', padding=5)
        self.button_frame.grid(row=3, column=1, sticky=(N, W, E, S), padx=(5, 0))

        # --- NEW: Statistics Frame ---
        self.stats_frame = ttk.Frame(self.main_frame, style='Grooved.TFrame', padding=5)
        self.stats_frame.grid(row=4, column=1, sticky=(N, W, E, S), padx=(5, 0), pady=5)

        self.main_frame.columnconfigure(1, weight=1)
        # Adjust row weights for new stats frame
        self.main_frame.rowconfigure(1, weight=1)  # detail_frame can expand
        self.main_frame.rowconfigure(2, weight=0)  # input_frame fixed
        self.main_frame.rowconfigure(3, weight=0)  # button_frame fixed
        self.main_frame.rowconfigure(4, weight=0)  # stats_frame fixed

        self.image_label = ttk.Label(self.image_frame,
                                     background=DARK_MODE_BG)  # Ensure image label bg matches image_frame
        self.image_label.pack(padx=5, pady=5, fill=BOTH, expand=True)
        self.image_frame.bind("<Configure>", self.on_frame_resize)

        img_list_label = ttk.Label(self.list_frame, text="Images:", style='TLabel')
        img_list_label.pack(padx=5, pady=(5, 0), anchor=W)
        self.image_listbox = Listbox(self.list_frame, height=10, exportselection=False,
                                     bg=DARK_MODE_WIDGET_BG, fg=DARK_MODE_TEXT_FG,
                                     selectbackground=DARK_MODE_SELECT_BG, selectforeground=DARK_MODE_SELECT_FG,
                                     highlightbackground=DARK_MODE_BORDER_COLOR, highlightcolor=DARK_MODE_SELECT_BG,
                                     relief='solid', borderwidth=1)
        self.image_listbox.pack(padx=5, pady=(0, 5), fill=X)
        self.image_listbox.bind("<<ListboxSelect>>", self.on_image_select)

        feat_list_label = ttk.Label(self.list_frame, text="Features:", style='TLabel')
        feat_list_label.pack(padx=5, pady=(5, 0), anchor=W)
        self.feature_listbox = Listbox(self.list_frame, height=10, exportselection=False,
                                       bg=DARK_MODE_WIDGET_BG, fg=DARK_MODE_TEXT_FG,
                                       selectbackground=DARK_MODE_SELECT_BG, selectforeground=DARK_MODE_SELECT_FG,
                                       highlightbackground=DARK_MODE_BORDER_COLOR, highlightcolor=DARK_MODE_SELECT_BG,
                                       relief='solid', borderwidth=1)
        self.feature_listbox.pack(padx=5, pady=(0, 5), fill=X)
        self.feature_listbox.bind("<<ListboxSelect>>", self.on_mask_select)

        self.selected_feature_label = ttk.Label(self.detail_frame, text="No Feature Selected", style='TLabel')
        self.selected_feature_label.pack(padx=5, pady=5, anchor=W)

        self.detail_label = ttk.Label(self.detail_frame, text="Attributes:", style='TLabel')
        self.detail_label.pack(padx=5, pady=(5, 0), anchor=W)
        self.attribute_text = Text(self.detail_frame, height=5, width=40, state=DISABLED, wrap=WORD,
                                   bg=DARK_MODE_WIDGET_BG, fg=DARK_MODE_TEXT_FG,
                                   selectbackground=DARK_MODE_SELECT_BG, selectforeground=DARK_MODE_SELECT_FG,
                                   insertbackground=DARK_MODE_INSERT_COLOR,  # Cursor color
                                   highlightbackground=DARK_MODE_BORDER_COLOR, highlightcolor=DARK_MODE_SELECT_BG,
                                   relief='solid', borderwidth=1)
        self.attribute_text.pack(padx=5, pady=(0, 5), fill=X)

        self.relationship_label_display = ttk.Label(self.detail_frame, text="Relationships (Click to Edit):",
                                                    style='TLabel')
        self.relationship_label_display.pack(padx=5, pady=(5, 0), anchor=W)
        self.relationship_listbox = Listbox(self.detail_frame, height=5, width=40, exportselection=False,
                                            bg=DARK_MODE_WIDGET_BG, fg=DARK_MODE_TEXT_FG,
                                            selectbackground=DARK_MODE_SELECT_BG, selectforeground=DARK_MODE_SELECT_FG,
                                            highlightbackground=DARK_MODE_BORDER_COLOR,
                                            highlightcolor=DARK_MODE_SELECT_BG,
                                            relief='solid', borderwidth=1)
        self.relationship_listbox.pack(padx=5, pady=(0, 5), fill=X)
        self.relationship_listbox.bind("<<ListboxSelect>>", self.on_relationship_list_select)

        self.input_label = ttk.Label(self.input_frame, text="Add/Edit Relationship:", style='Input.TLabel')
        self.input_label.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky=W)

        rel_label_prompt = ttk.Label(self.input_frame, text="Label:", style='Input.TLabel')
        rel_label_prompt.grid(row=1, column=0, padx=(5, 0), pady=5, sticky=W)
        self.relationship_label_var = StringVar()
        self.relationship_label_entry = ttk.Entry(self.input_frame, textvariable=self.relationship_label_var, width=20,
                                                  style='TEntry')
        self.relationship_label_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky=(W, E))
        self.setup_autocomplete()

        target_label_prompt = ttk.Label(self.input_frame, text="Target:", style='Input.TLabel')
        target_label_prompt.grid(row=2, column=0, padx=(5, 0), pady=5, sticky=W)
        self.target_mask_id_var = StringVar()
        self.target_mask_id_combobox = ttk.Combobox(self.input_frame, textvariable=self.target_mask_id_var, width=18,
                                                    state="readonly", style='TCombobox')
        self.target_mask_id_combobox.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky=(W, E))
        self.target_mask_id_combobox.bind("<<ComboboxSelected>>", self.on_target_mask_select)

        self.add_relationship_button = ttk.Button(self.input_frame, text="Add", command=self.on_add_relationship,
                                                  style='TButton')
        self.add_relationship_button.grid(row=3, column=0, padx=5, pady=5, sticky=W)
        self.edit_relationship_button = ttk.Button(self.input_frame, text="Update", command=self.on_edit_relationship,
                                                   style='TButton')
        self.edit_relationship_button.grid(row=3, column=1, padx=5, pady=5, sticky=W)
        self.delete_relationship_button = ttk.Button(self.input_frame, text="Delete",
                                                     command=self.on_delete_relationship, style='TButton')
        self.delete_relationship_button.grid(row=3, column=2, padx=5, pady=5, sticky=W)
        self.input_frame.columnconfigure(1, weight=1)

        self.load_button = ttk.Button(self.button_frame, text="Load XML", command=self.load_xml, style='TButton')
        self.load_button.pack(side=LEFT, padx=5, pady=5)
        self.save_button = ttk.Button(self.button_frame, text="Save XML", command=self.save_xml, style='TButton')
        self.save_button.pack(side=LEFT, padx=5, pady=5)
        self.load_images_button = ttk.Button(self.button_frame, text="Load Images", command=self.load_images,
                                             style='TButton')
        self.load_images_button.pack(side=LEFT, padx=5, pady=5)
        self.undo_button = ttk.Button(self.button_frame, text="Undo", command=self.undo, state=DISABLED,
                                      style='TButton')
        self.undo_button.pack(side=LEFT, padx=5, pady=5)
        self.redo_button = ttk.Button(self.button_frame, text="Redo", command=self.redo, state=DISABLED,
                                      style='TButton')
        self.redo_button.pack(side=LEFT, padx=5, pady=5)

        # --- NEW: Statistics Widgets ---
        stats_title_label = ttk.Label(self.stats_frame, text="Project Statistics", style='TLabel',
                                      font=('TkDefaultFont', 10, 'bold'))
        stats_title_label.grid(row=0, column=0, columnspan=2, sticky=W, padx=5, pady=(5, 2))

        self.total_images_var = StringVar(value="Images: 0")
        self.total_features_var = StringVar(value="Features: 0")
        self.total_attributes_var = StringVar(value="Attributes: 0")
        self.total_relationships_var = StringVar(value="Relationships: 0")

        ttk.Label(self.stats_frame, textvariable=self.total_images_var, style='Stats.TLabel').grid(row=1, column=0,
                                                                                                   sticky=W, padx=5,
                                                                                                   pady=1)
        ttk.Label(self.stats_frame, textvariable=self.total_features_var, style='Stats.TLabel').grid(row=1, column=1,
                                                                                                     sticky=W, padx=5,
                                                                                                     pady=1)
        ttk.Label(self.stats_frame, textvariable=self.total_attributes_var, style='Stats.TLabel').grid(row=2, column=0,
                                                                                                       sticky=W, padx=5,
                                                                                                       pady=1)
        ttk.Label(self.stats_frame, textvariable=self.total_relationships_var, style='Stats.TLabel').grid(row=2,
                                                                                                          column=1,
                                                                                                          sticky=W,
                                                                                                          padx=5,
                                                                                                          pady=1)
        self.stats_frame.columnconfigure((0, 1), weight=1)

        self.menubar = Menu(self.root)  # Will inherit from root.option_add
        self.filemenu = Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Open XML", command=self.load_xml, accelerator="Ctrl+O")
        self.filemenu.add_command(label="Load Images", command=self.load_images, accelerator="Ctrl+L")
        self.filemenu.add_command(label="Save XML", command=self.save_xml, accelerator="Ctrl+S")
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.root.quit)
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.editmenu = Menu(self.menubar, tearoff=0)
        self.editmenu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z", state=DISABLED)
        self.editmenu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y", state=DISABLED)
        self.menubar.add_cascade(label="Edit", menu=self.editmenu)
        self.root.config(menu=self.menubar)

        self.root.bind_all("<Control-o>", lambda event: self.load_xml())
        self.root.bind_all("<Control-l>", lambda event: self.load_images())
        self.root.bind_all("<Control-s>", lambda event: self.save_xml())
        self.root.bind_all("<Control-z>", lambda event: self.undo())
        self.root.bind_all("<Control-y>", lambda event: self.redo())

        self.status_label = ttk.Label(self.root, text="Ready", relief=SUNKEN, anchor=W, style='Status.TLabel')
        self.status_label.grid(row=1, column=0, sticky=(W, E))

        self.update_button_states()
        self.update_statistics()  # Initial call

    def update_statistics(self):
        """Calculates and updates the statistics display panel."""
        if not self.image_data:
            self.total_images_var.set("Images: 0")
            self.total_features_var.set("Features: 0")
            self.total_attributes_var.set("Attributes: 0")
            self.total_relationships_var.set("Relationships: 0")
            return

        num_images = len(self.image_data)
        num_features = 0
        num_attributes = 0
        num_relationships = 0

        for img_data in self.image_data.values():
            num_features += len(img_data.get("masks", []))
            for mask in img_data.get("masks", []):
                num_attributes += len(mask.get("attributes", {}))
                num_relationships += len(mask.get("relationships", []))

        self.total_images_var.set(f"Images: {num_images}")
        self.total_features_var.set(f"Features: {num_features}")
        self.total_attributes_var.set(f"Attributes: {num_attributes}")
        self.total_relationships_var.set(f"Relationships: {num_relationships}")

    def update_button_states(self):
        if self.relationship_listbox.curselection():
            self.add_relationship_button.config(state=DISABLED)
            self.edit_relationship_button.config(state=NORMAL)
            self.delete_relationship_button.config(state=NORMAL)
        else:
            self.add_relationship_button.config(state=NORMAL)
            self.edit_relationship_button.config(state=DISABLED)
            self.delete_relationship_button.config(state=DISABLED)

    def on_frame_resize(self, event=None):
        if self._resize_timer:
            self.root.after_cancel(self._resize_timer)
        self._resize_timer = self.root.after(100, self.redraw_image_with_masks)

    def load_xml(self):
        file_path = filedialog.askopenfilename(filetypes=[("XML files", "*.xml")])
        if file_path:
            try:
                self.loaded_xml_path = file_path
                self.image_data = self.load_cvat_xml(file_path)
                self.populate_image_list()
                if self.image_listbox.size() > 0:
                    self.image_listbox.selection_set(0)
                    self.on_image_select(None)
                else:
                    self.image_counter_label.config(text="Image 0/0")  # Handle empty xml
                self.status_label.config(text=f"Loaded XML: {os.path.basename(file_path)}")
                self.undo_stack = []
                self.redo_stack = []
                self.update_undo_redo_buttons()
                self.update_statistics()  # Update stats after loading
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load XML: {e}\n\nCheck XML format and file permissions.")
                self.status_label.config(text="Error loading XML")
                traceback.print_exc()
                self.loaded_xml_path = None
                self.image_data = {}
                self.clear_all_lists_and_details()

    def load_images(self):
        if not self.image_data:
            messagebox.showwarning("Warning", "Please load the XML annotation file first.")
            return
        dir_path = filedialog.askdirectory(title="Select Image Directory")
        if dir_path:
            self.image_directory = dir_path
            if self.current_image:
                image_path = os.path.join(self.image_directory, self.current_image)
                if os.path.exists(image_path):
                    self.redraw_image_with_masks()
                    self.status_label.config(text=f"Image directory: {self.image_directory}")
                else:
                    messagebox.showwarning("Image Not Found",
                                           f"Current image '{self.current_image}' not found in the selected directory.\nPlease select the image from the list again if needed.")
                    self.image_label.config(image=None)
                    self.tk_image = None
                    self.status_label.config(text=f"Image directory set, but '{self.current_image}' not found.")
            else:
                self.status_label.config(text=f"Image directory set: {self.image_directory}. Select an image to view.")

    def save_xml(self):
        if not self.image_data or not self.loaded_xml_path:
            messagebox.showerror("Error", "No XML data loaded or loaded path is missing.")
            return
        original_dir = os.path.dirname(self.loaded_xml_path)
        original_filename = os.path.basename(self.loaded_xml_path)
        name, ext = os.path.splitext(original_filename)
        suggested_filename = f"{name}_relationships{ext}"
        initial_file = os.path.join(original_dir, suggested_filename)
        file_path = filedialog.asksaveasfilename(
            initialfile=initial_file,
            defaultextension=".xml",
            filetypes=[("XML files", "*.xml")],
            title="Save Annotations As")
        if file_path:
            try:
                self.update_xml_with_relationships(self.image_data, self.loaded_xml_path, file_path)
                self.save_relationship_labels()
                self.status_label.config(text=f"Saved XML to: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Annotations saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save XML: {e}")
                traceback.print_exc()
                self.status_label.config(text="Error saving XML")

    def load_cvat_xml(self, xml_file):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            image_data = {}
        except ET.ParseError as e:
            print(f"XML Parse Error loading '{xml_file}': {e}")
            raise
        for image_elem in root.findall("image"):
            image_filename = image_elem.get("name")
            if not image_filename:
                print("Warning: Skipping image element with no 'name' attribute.")
                continue
            image_width = int(image_elem.get("width", 0))
            image_height = int(image_elem.get("height", 0))
            image_id_from_xml = image_elem.get("id")
            if image_width <= 0 or image_height <= 0:
                print(
                    f"Warning: Skipping image '{image_filename}' due to invalid dimensions ({image_width}x{image_height}).")
                continue
            image_data[image_filename] = {
                "width_original": image_width,
                "height_original": image_height,
                "masks": [],
                "relationships_from_xml": [],
                "id_from_xml": image_id_from_xml
            }
            mask_internal_id_counter = 1
            annotation_elements_in_xml_order = [elem for elem in image_elem.findall("*") if
                                                elem.tag in ["mask", "box", "polygon", "tag"]]

            for elem in annotation_elements_in_xml_order:
                internal_id = str(mask_internal_id_counter)
                mask_data = {
                    "id": internal_id,
                    "label": elem.get("label", "Unknown"),
                    "attributes": {},
                    "relationships": [],
                    "elem_type": elem.tag,
                    "xml_element": elem
                }
                for attr_elem in elem.findall("attribute"):
                    attr_name = attr_elem.get("name")
                    if attr_name:
                        mask_data["attributes"][attr_name] = attr_elem.text

                for rel_elem in elem.findall("relationship"):
                    target_xml_id = rel_elem.get("with")
                    rel_label_text = rel_elem.text
                    if target_xml_id and rel_label_text:
                        image_data[image_filename]["relationships_from_xml"].append({
                            "source_internal_id_placeholder": internal_id,
                            "target_xml_id": target_xml_id,
                            "label": rel_label_text.strip()
                        })
                bbox_coords = None
                if elem.tag == "mask":
                    mask_data["rle"] = elem.get("rle")
                    try:
                        left = float(elem.get("left", 0))
                    except ValueError:
                        left = 0.0
                    try:
                        top = float(elem.get("top", 0))
                    except ValueError:
                        top = 0.0
                    try:
                        width = float(elem.get("width", 0))
                    except ValueError:
                        width = 0.0
                    try:
                        height = float(elem.get("height", 0))
                    except ValueError:
                        height = 0.0
                    if width > 0 and height > 0: bbox_coords = (left, top, width, height)
                elif elem.tag == "box":
                    try:
                        xtl = float(elem.get("xtl", 0))
                    except ValueError:
                        xtl = 0.0
                    try:
                        ytl = float(elem.get("ytl", 0))
                    except ValueError:
                        ytl = 0.0
                    try:
                        xbr = float(elem.get("xbr", 0))
                    except ValueError:
                        xbr = 0.0
                    try:
                        ybr = float(elem.get("ybr", 0))
                    except ValueError:
                        ybr = 0.0
                    width = xbr - xtl
                    height = ybr - ytl
                    if width > 0 and height > 0: bbox_coords = (xtl, ytl, width, height)
                elif elem.tag == "polygon":
                    points_str = elem.get("points")
                    if points_str:
                        try:
                            points = [float(p) for p in points_str.replace(';', ',').split(",")]
                            mask_data["points"] = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
                            if mask_data["points"]:
                                xs = [p[0] for p in mask_data["points"]]
                                ys = [p[1] for p in mask_data["points"]]
                                min_x, max_x = min(xs), max(xs)
                                min_y, max_y = min(ys), max(ys)
                                width = max_x - min_x
                                height = max_y - min_y
                                if width > 0 and height > 0: bbox_coords = (min_x, min_y, width, height)
                        except (ValueError, IndexError):
                            print(f"Warning: Could not parse points for polygon in {image_filename}: {points_str}")
                            mask_data["points"] = []
                if bbox_coords and bbox_coords[2] > 0 and bbox_coords[3] > 0:
                    mask_data["bbox"] = bbox_coords
                else:
                    mask_data["bbox"] = None
                image_data[image_filename]["masks"].append(mask_data)
                mask_internal_id_counter += 1

        for img_fname, data in image_data.items():
            if "relationships_from_xml" in data and data["relationships_from_xml"]:
                for rel_from_xml in data["relationships_from_xml"]:
                    source_internal_id = rel_from_xml["source_internal_id_placeholder"]
                    target_xml_id_from_file = rel_from_xml["target_xml_id"]
                    source_mask_object = next((m for m in data["masks"] if m["id"] == source_internal_id), None)
                    if not source_mask_object:
                        continue
                    target_internal_id = None
                    try:
                        potential_target_mask = next((m for m in data["masks"] if m["id"] == target_xml_id_from_file),
                                                     None)
                        if potential_target_mask:
                            target_internal_id = potential_target_mask["id"]
                    except ValueError:
                        pass
                    if target_internal_id:
                        source_mask_object["relationships"].append({
                            "label": rel_from_xml["label"],
                            "target_id": target_internal_id
                        })
                    else:
                        print(
                            f"Warning: Could not map target XML ID '{target_xml_id_from_file}' to an internal ID for relationship '{rel_from_xml['label']}' in image '{img_fname}'.")
        return image_data

    def update_xml_with_relationships(self, image_data_param, original_xml_file, output_xml_file):
        try:
            tree = ET.parse(original_xml_file)
            root = tree.getroot()
        except ET.ParseError as e:
            messagebox.showerror("XML Parse Error",
                                 f"Error parsing original XML file '{os.path.basename(original_xml_file)}' for saving: {e}")
            raise
        for image_elem_in_new_tree in root.findall("image"):
            image_filename = image_elem_in_new_tree.get("name")
            if image_filename not in image_data_param:
                print(
                    f"Warning: Image '{image_filename}' from XML not found in internal data during save. Skipping this image element.")
                continue
            current_image_internal_data = image_data_param[image_filename]
            internal_id_to_saved_xml_id_map = {}
            internal_id_to_new_xml_element_map = {}
            annotation_xml_elements_in_new_tree = [
                elem for elem in image_elem_in_new_tree.findall("*")
                if elem.tag in ["mask", "box", "polygon", "tag"]
            ]
            if len(annotation_xml_elements_in_new_tree) != len(current_image_internal_data["masks"]):
                print(
                    f"Warning: Mismatch in count of annotation elements in XML ({len(annotation_xml_elements_in_new_tree)}) " +
                    f"vs internal data ({len(current_image_internal_data['masks'])}) for {image_filename}. " +
                    "Saving will proceed but might be incomplete or incorrect for this image.")
            loop_limit = min(len(annotation_xml_elements_in_new_tree), len(current_image_internal_data["masks"]))
            saved_xml_id_counter = 1
            for i in range(loop_limit):
                internal_mask_item_data = current_image_internal_data["masks"][i]
                xml_element_to_modify = annotation_xml_elements_in_new_tree[i]
                current_saved_xml_id = str(saved_xml_id_counter)
                xml_element_to_modify.set("id", current_saved_xml_id)
                internal_id_to_saved_xml_id_map[internal_mask_item_data['id']] = current_saved_xml_id
                internal_id_to_new_xml_element_map[internal_mask_item_data['id']] = xml_element_to_modify
                saved_xml_id_counter += 1
            if len(current_image_internal_data["masks"]) > len(annotation_xml_elements_in_new_tree):
                for i in range(len(annotation_xml_elements_in_new_tree), len(current_image_internal_data["masks"])):
                    unmapped_internal_item = current_image_internal_data["masks"][i]
                    print(
                        f"Warning: Internal mask item with ID '{unmapped_internal_item['id']}' in {image_filename} had no corresponding XML element to save to.")
            for internal_mask_item_data in current_image_internal_data["masks"]:
                source_xml_element_new_tree = internal_id_to_new_xml_element_map.get(internal_mask_item_data['id'])
                if not ET.iselement(source_xml_element_new_tree):
                    if internal_mask_item_data.get("relationships"):
                        print(
                            f"Warning: Skipping relationships for internal mask ID '{internal_mask_item_data['id']}' in {image_filename} " +
                            "as its XML element in the new tree was not mapped.")
                    continue
                for rel_tag in source_xml_element_new_tree.findall("relationship"):
                    source_xml_element_new_tree.remove(rel_tag)
                if "relationships" in internal_mask_item_data:
                    for relationship_detail in internal_mask_item_data["relationships"]:
                        target_internal_id = relationship_detail["target_id"]
                        relationship_label_text = relationship_detail["label"]
                        target_saved_xml_id = internal_id_to_saved_xml_id_map.get(target_internal_id)
                        if target_saved_xml_id:
                            rel_elem_new = ET.SubElement(source_xml_element_new_tree, "relationship")
                            rel_elem_new.set("with", target_saved_xml_id)
                            rel_elem_new.text = relationship_label_text
                        else:
                            source_element_xml_id_for_warning = source_xml_element_new_tree.get('id',
                                                                                                internal_mask_item_data[
                                                                                                    'id'])
                            print(
                                f"Warning: Could not find saved XML ID for target internal ID '{target_internal_id}'. " +
                                f"Relationship '{relationship_label_text}' from source XML ID '{source_element_xml_id_for_warning}' " +
                                f"(internal ID '{internal_mask_item_data['id']}') in {image_filename} will not be saved.")
        try:
            import xml.dom.minidom
            xml_str = ET.tostring(root, encoding='unicode')
            dom = xml.dom.minidom.parseString(xml_str)
            pretty_xml_as_string = dom.toprettyxml(indent="  ", encoding="utf-8")
            with open(output_xml_file, "wb") as f:
                f.write(pretty_xml_as_string)
        except ImportError:
            print("Warning: xml.dom.minidom not found. Saving XML without pretty printing.")
            tree.write(output_xml_file, encoding="utf-8", xml_declaration=True)
        except Exception as e:
            messagebox.showerror("XML Write Error", f"Error during XML writing/pretty printing: {e}")
            traceback.print_exc()
            try:
                print("Attempting fallback XML write...")
                tree.write(output_xml_file, encoding="utf-8", xml_declaration=True)
            except Exception as write_e:
                messagebox.showerror("XML Write Error", f"Fallback XML write also failed: {write_e}")
                raise write_e

    def populate_image_list(self):
        self.image_listbox.delete(0, END)
        # --- MODIFIED: Add numbers to image list ---
        for i, image_filename in enumerate(sorted(self.image_data.keys())):
            self.image_listbox.insert(END, f"{i + 1}) {image_filename}")

    def populate_feature_list(self):
        self.feature_listbox.delete(0, END)
        if self.current_image and self.current_image in self.image_data:
            try:
                sorted_masks = sorted(self.image_data[self.current_image]["masks"], key=lambda m: int(m['id']))
            except (ValueError, TypeError):
                print("Warning: Non-integer mask ID found during sorting. Sorting may be inconsistent.")
                sorted_masks = sorted(self.image_data[self.current_image]["masks"], key=lambda m: m['id'])
            for mask_item in sorted_masks:
                self.feature_listbox.insert(END, f"{mask_item['id']}:{mask_item['label']}")
        self.clear_relationship_input()
        self.selected_mask_id = None
        self.display_selected_mask_details()

    def populate_feature_dropdown(self):
        current_values = []
        self._target_mask_map = {}
        if self.current_image and self.current_image in self.image_data:
            try:
                sorted_masks = sorted(self.image_data[self.current_image]["masks"], key=lambda m: int(m['id']))
            except (ValueError, TypeError):
                sorted_masks = sorted(self.image_data[self.current_image]["masks"], key=lambda m: m['id'])
            for mask_item in sorted_masks:
                if mask_item['id'] != self.selected_mask_id:
                    display_text = f"{mask_item['id']}:{mask_item['label']}"
                    current_values.append(display_text)
                    self._target_mask_map[display_text] = mask_item['id']
        self.target_mask_id_combobox['values'] = current_values
        current_selection = self.target_mask_id_var.get()
        if current_selection not in self._target_mask_map:
            self.target_mask_id_combobox.set('')
            self.selected_target_mask_id = None

    def redraw_image_with_masks(self):
        if not self.current_image or not self.image_directory:
            self.image_label.config(image=None);
            self.tk_image = None;
            return
        image_path = os.path.join(self.image_directory, self.current_image)
        if not os.path.exists(image_path):
            self.image_label.config(image=None);
            self.tk_image = None;
            return
        source_mask_to_draw = self.get_mask_by_id(self.current_image,
                                                  self.selected_mask_id) if self.selected_mask_id else None
        target_mask_to_draw = self.get_mask_by_id(self.current_image,
                                                  self.selected_target_mask_id) if self.selected_target_mask_id else None
        self.draw_masks_on_image(
            image_path,
            [source_mask_to_draw] if source_mask_to_draw else [],
            [target_mask_to_draw] if target_mask_to_draw else []
        )

    def get_mask_by_id(self, image_filename, mask_id_param):
        if image_filename in self.image_data:
            for mask_item in self.image_data[image_filename]["masks"]:
                if mask_item["id"] == mask_id_param:
                    return mask_item
        return None

    def decode_rle(self, rle_string, width, height):
        if not rle_string or width <= 0 or height <= 0: return np.zeros((height, width), dtype=np.uint8)
        try:
            rle_list = list(map(int, rle_string.split(',')))
        except ValueError:
            print(f"Warn: Invalid RLE: {rle_string[:50]}...");
            return np.zeros((height, width), dtype=np.uint8)
        mask_flat = np.zeros(width * height, dtype=np.uint8);
        index = 0;
        value = 0
        for count in rle_list:
            end_index = index + count
            if value == 1:
                start = min(max(0, index), len(mask_flat));
                end = min(max(0, end_index), len(mask_flat))
                if start < end: mask_flat[start:end] = 1
            if end_index > len(mask_flat): index = len(mask_flat); break
            index = end_index;
            value = 1 - value
        try:
            return mask_flat.reshape((height, width))
        except ValueError:
            print(f"Error: RLE reshape failed {len(mask_flat)} vs {height}x{width}");
            return np.zeros((height, width),
                            dtype=np.uint8)

    def draw_masks_on_image(self, image_path, source_masks, target_masks=None):
        if target_masks is None: target_masks = []
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None: print(f"Error reading {image_path}"); self.image_label.config(
                image=None); self.tk_image = None; return
            if self.current_image not in self.image_data: return
            img_meta = self.image_data.get(self.current_image)
            if not img_meta or "height_original" not in img_meta or "width_original" not in img_meta:
                print(f"Error: Missing original image dimensions for {self.current_image}");
                return
            orig_h, orig_w = img_meta["height_original"], img_meta["width_original"]
            disp_w, disp_h = self.image_frame.winfo_width(), self.image_frame.winfo_height()
            if orig_w <= 0 or orig_h <= 0 or disp_w <= 1 or disp_h <= 1: return
            img_display_bgr = cv2.resize(img_bgr, (disp_w, disp_h))
            color_overlay = np.zeros_like(img_display_bgr, dtype=img_display_bgr.dtype)
            for mask_list, mask_color_val, bbox_draw_color_val in [(source_masks, SOURCE_MASK_COLOR, SOURCE_BBOX_COLOR),
                                                                   (target_masks, TARGET_MASK_COLOR,
                                                                    TARGET_BBOX_COLOR)]:
                for mask_item_detail in mask_list:
                    if mask_item_detail is None: continue
                    bbox = mask_item_detail.get('bbox')
                    if mask_item_detail.get('rle') and bbox:
                        rle_str = mask_item_detail['rle']
                        try:
                            left_f, top_f, bbox_w_f, bbox_h_f = map(float, bbox)
                            left_i, top_i, bbox_w_i, bbox_h_i = int(left_f), int(top_f), int(bbox_w_f), int(bbox_h_f)
                        except (ValueError, TypeError):
                            continue
                        if bbox_w_i > 0 and bbox_h_i > 0:
                            decoded_bbox_mask = self.decode_rle(rle_str, bbox_w_i, bbox_h_i)
                            if decoded_bbox_mask.shape == (bbox_h_i, bbox_w_i):
                                full_mask_original_res = np.zeros((orig_h, orig_w), dtype=np.uint8)
                                y_start_coord, y_end_coord = max(0, top_i), min(orig_h, top_i + bbox_h_i)
                                x_start_coord, x_end_coord = max(0, left_i), min(orig_w, left_i + bbox_w_i)
                                bbox_y_start_slice, bbox_y_end_slice = y_start_coord - top_i, y_end_coord - top_i
                                bbox_x_start_slice, bbox_x_end_slice = x_start_coord - left_i, x_end_coord - left_i
                                if y_end_coord > y_start_coord and x_end_coord > x_start_coord and \
                                        bbox_y_end_slice > bbox_y_start_slice and bbox_x_end_slice > bbox_x_start_slice:
                                    try:
                                        bbox_mask_slice = decoded_bbox_mask[bbox_y_start_slice:bbox_y_end_slice,
                                                          bbox_x_start_slice:bbox_x_end_slice]
                                        target_img_slice = full_mask_original_res[y_start_coord:y_end_coord,
                                                           x_start_coord:x_end_coord]
                                        if target_img_slice.shape == bbox_mask_slice.shape:
                                            target_img_slice[:] = bbox_mask_slice
                                        else:
                                            target_h_val, target_w_val = target_img_slice.shape
                                            if bbox_mask_slice.size > 0 and target_h_val > 0 and target_w_val > 0:
                                                resized_bbox_slice = cv2.resize(bbox_mask_slice,
                                                                                (target_w_val, target_h_val),
                                                                                interpolation=cv2.INTER_NEAREST)
                                                target_img_slice[:] = resized_bbox_slice
                                    except Exception as slice_err:
                                        print(
                                            f"Error during RLE mask slice assignment for mask {mask_item_detail.get('id')}: {slice_err}")
                                full_mask_display_res = cv2.resize(full_mask_original_res, (disp_w, disp_h),
                                                                   interpolation=cv2.INTER_NEAREST)
                                color_overlay[full_mask_display_res == 1] = mask_color_val
                    if bbox:
                        try:
                            left_f_bbox, top_f_bbox, width_f_bbox, height_f_bbox = map(float, bbox)
                        except (ValueError, TypeError):
                            continue
                        if width_f_bbox > 0 and height_f_bbox > 0:
                            scale_x_factor, scale_y_factor = disp_w / orig_w, disp_h / orig_h
                            l_px_coord, t_px_coord = int(left_f_bbox * scale_x_factor), int(top_f_bbox * scale_y_factor)
                            r_px_coord, b_px_coord = int((left_f_bbox + width_f_bbox) * scale_x_factor), int(
                                (top_f_bbox + height_f_bbox) * scale_y_factor)
                            cv2.rectangle(img_display_bgr, (l_px_coord, t_px_coord), (r_px_coord, b_px_coord),
                                          bbox_draw_color_val, 2)
            alpha_blend = 0.4
            active_mask_indices = np.any(color_overlay > 0, axis=-1)
            if np.any(active_mask_indices):
                src1_pixels = color_overlay[active_mask_indices].astype(img_display_bgr.dtype)
                src2_pixels = img_display_bgr[active_mask_indices].astype(img_display_bgr.dtype)
                img_display_bgr[active_mask_indices] = cv2.addWeighted(src1_pixels, float(alpha_blend), src2_pixels,
                                                                       float(1.0 - alpha_blend), float(0))
            img_rgb_final = cv2.cvtColor(img_display_bgr, cv2.COLOR_BGR2RGB)
            img_pil_final = Image.fromarray(img_rgb_final)
            self.tk_image = ImageTk.PhotoImage(img_pil_final)
            self.image_label.config(image=self.tk_image)
        except Exception as e:
            print(f"--- Error during mask/image drawing for {self.current_image} ---");
            traceback.print_exc()
            self.image_label.config(image=None);
            self.tk_image = None

    def display_selected_mask_details(self):
        self.attribute_text.config(state=NORMAL);
        self.attribute_text.delete("1.0", END)
        self.relationship_listbox.delete(0, END)
        mask_details_data = self.get_mask_by_id(self.current_image,
                                                self.selected_mask_id) if self.current_image and self.selected_mask_id else None
        if mask_details_data:
            self.selected_feature_label.config(text=f"Selected: {mask_details_data['id']}:{mask_details_data['label']}")
            attrs_str = "\n".join(f"{k}: {v}" for k, v in sorted(mask_details_data["attributes"].items()));
            self.attribute_text.insert(END, attrs_str if attrs_str else "No attributes")
            if mask_details_data["relationships"]:
                current_img_masks = self.image_data.get(self.current_image, {}).get("masks", [])
                id_to_label_map = {m['id']: m['label'] for m in current_img_masks}
                try:
                    sorted_rels_list = sorted(mask_details_data["relationships"],
                                              key=lambda r: (r['label'], int(r['target_id'])))
                except (ValueError, TypeError):
                    sorted_rels_list = sorted(mask_details_data["relationships"],
                                              key=lambda r: (r['label'], r['target_id']))
                for r_item in sorted_rels_list:
                    target_label_str = id_to_label_map.get(r_item['target_id'], 'Unknown Label')
                    self.relationship_listbox.insert(END,
                                                     f"{r_item['label']} -> {r_item['target_id']}:{target_label_str}")
            else:
                self.relationship_listbox.insert(END, "(No relationships)")
        else:
            self.selected_feature_label.config(text="No Feature Selected");
            self.attribute_text.insert(END, "N/A");
            self.relationship_listbox.insert(END, "N/A")
        self.attribute_text.config(state=DISABLED)
        self.clear_relationship_input()
        self.update_statistics()  # Update stats on selection change

    def clear_all_lists_and_details(self):
        self.image_listbox.delete(0, END);
        self.feature_listbox.delete(0, END);
        self.relationship_listbox.delete(0, END)
        self.attribute_text.config(state=NORMAL);
        self.attribute_text.delete('1.0', END);
        self.attribute_text.config(state=DISABLED)
        self.selected_feature_label.config(text="No Feature Selected");
        self.clear_relationship_input()
        self.image_label.config(image=None);
        self.tk_image = None;
        self.current_image = None;
        self.selected_mask_id = None
        self.image_counter_label.config(text="Image 0/0")
        self.update_statistics()  # Reset stats

    def on_image_select(self, event):
        sel_indices = self.image_listbox.curselection()
        if sel_indices:
            idx = sel_indices[0]
            # --- MODIFIED: Parse filename from "1) filename.ext" format ---
            selected_text = self.image_listbox.get(idx)
            new_img_filename = selected_text.split(") ", 1)[1]

            # --- MODIFIED: Update image counter ---
            total_images = self.image_listbox.size()
            self.image_counter_label.config(text=f"Image {idx + 1}/{total_images}")

            if new_img_filename != self.current_image:
                self.current_image = new_img_filename;
                self.populate_feature_list();
                self.selected_mask_id = None;
                self.clear_relationship_input();
                self.populate_feature_dropdown();
                self.display_selected_mask_details();
                self.redraw_image_with_masks()

    def on_mask_select(self, event):
        sel_indices = self.feature_listbox.curselection()
        if sel_indices:
            selected_text = self.feature_listbox.get(sel_indices[0]);
            new_selected_id = selected_text.split(":", 1)[0]
            if new_selected_id != self.selected_mask_id:
                self.selected_mask_id = new_selected_id;
                self.clear_relationship_input();
                self.populate_feature_dropdown();
                self.display_selected_mask_details();
                self.redraw_image_with_masks()

    def on_target_mask_select(self, event=None):
        selected_text = self.target_mask_id_var.get()
        new_target_id_val = self._target_mask_map.get(selected_text) if hasattr(self, '_target_mask_map') else None
        if new_target_id_val != self.selected_target_mask_id:
            self.selected_target_mask_id = new_target_id_val;
            self.redraw_image_with_masks()

    def on_relationship_list_select(self, event=None):
        sel_indices = self.relationship_listbox.curselection()
        if not sel_indices: self.update_button_states(); return
        selected_rel_text = self.relationship_listbox.get(sel_indices[0])
        try:
            label_text_part, target_text_part = selected_rel_text.split(" -> ", 1)
            target_id_str, target_label_from_list = target_text_part.split(":", 1)
            target_display_text = f"{target_id_str}:{target_label_from_list.strip()}"
            self.relationship_label_var.set(label_text_part.strip())
            if target_display_text in self._target_mask_map:
                self.target_mask_id_var.set(target_display_text)
            else:
                found_by_id_flag = False
                for display_val, t_id_val in self._target_mask_map.items():
                    if t_id_val == target_id_str:
                        self.target_mask_id_var.set(display_val)
                        found_by_id_flag = True;
                        break
                if not found_by_id_flag: self.target_mask_id_var.set("")
            self.on_target_mask_select()
        except ValueError:
            print(f"Error parsing relationship selection from listbox: {selected_rel_text}")
        finally:
            self.update_button_states()

    def setup_autocomplete(self):
        entry_widget = self.relationship_label_entry
        entry_widget.bind("<KeyRelease>", self._on_autocomplete_keyrelease)
        entry_widget.bind("<Down>", self._on_autocomplete_down_arrow)
        entry_widget.bind("<Return>", self._on_autocomplete_select_with_enter)
        entry_widget.bind("<Escape>", self._on_autocomplete_escape)
        entry_widget.bind("<FocusOut>", self._on_autocomplete_focusout)

    def _on_autocomplete_select_with_enter(self, event):
        if self.autocomplete_listbox and self.autocomplete_listbox.winfo_exists():
            return self._on_autocomplete_select(event)
        return

    def _update_autocomplete_listbox(self):
        entry_widget = self.relationship_label_entry;
        current_text = self.relationship_label_var.get()
        matched_labels = sorted([lbl for lbl in self.relationship_labels if current_text.lower() in lbl.lower()])
        if not current_text or not matched_labels:
            self._destroy_autocomplete_listbox();
            return
        if not self.autocomplete_listbox or not self.autocomplete_listbox.winfo_exists():
            self.autocomplete_listbox = Listbox(self.input_frame,
                                                height=min(len(matched_labels), 6),
                                                width=entry_widget.winfo_width(),
                                                bg=DARK_MODE_WIDGET_BG, fg=DARK_MODE_TEXT_FG,
                                                # Dark theme for autocomplete
                                                selectbackground=DARK_MODE_SELECT_BG,
                                                selectforeground=DARK_MODE_SELECT_FG,
                                                highlightthickness=1, highlightbackground=DARK_MODE_BORDER_COLOR,
                                                relief='solid',
                                                borderwidth=0)  # Use borderwidth 0 if highlightthickness is used
            self.autocomplete_listbox.place(x=entry_widget.winfo_x(),
                                            y=entry_widget.winfo_y() + entry_widget.winfo_height(),
                                            anchor=NW)
            self.autocomplete_listbox.lift()
            self.autocomplete_listbox.bind("<ButtonRelease-1>", self._on_autocomplete_select)
            self.autocomplete_listbox.bind("<Return>", self._on_autocomplete_select)
            self.autocomplete_listbox.bind("<Tab>", self._on_autocomplete_select)
            self.autocomplete_listbox.bind("<Escape>", self._on_autocomplete_escape)
        self.autocomplete_listbox.delete(0, END)
        for label_item in matched_labels:
            self.autocomplete_listbox.insert(END, label_item)

    def _on_autocomplete_keyrelease(self, event):
        if event.keysym in ('Shift_L', 'Shift_R', 'Control_L', 'Control_R',
                            'Alt_L', 'Alt_R', 'Up', 'Down', 'Escape',
                            'Tab', 'Return', 'Enter'):
            return
        self._update_autocomplete_listbox()

    def _on_autocomplete_down_arrow(self, event):
        if self.autocomplete_listbox and self.autocomplete_listbox.winfo_exists():
            self.autocomplete_listbox.focus_set()
            self.autocomplete_listbox.selection_set(0)
            self.autocomplete_listbox.activate(0)
            return "break"
        return

    def _on_autocomplete_select(self, event):
        if not self.autocomplete_listbox or not self.autocomplete_listbox.winfo_exists(): return
        selected_indices = self.autocomplete_listbox.curselection()
        if selected_indices:
            selected_value = self.autocomplete_listbox.get(selected_indices[0])
            self.relationship_label_var.set(selected_value)
            self._destroy_autocomplete_listbox()
            self.relationship_label_entry.focus_set()
            self.relationship_label_entry.icursor(END)
            return "break"
        return

    def _on_autocomplete_escape(self, event=None):
        if self.autocomplete_listbox and self.autocomplete_listbox.winfo_exists():
            self._destroy_autocomplete_listbox()
            return "break"
        return

    def _on_autocomplete_focusout(self, event=None):
        self.root.after(150, self._check_and_destroy_autocomplete_on_focus_lost)

    def _check_and_destroy_autocomplete_on_focus_lost(self):
        if not self.autocomplete_listbox or not self.autocomplete_listbox.winfo_exists():
            return
        try:
            focused_widget = self.root.focus_get()
            if focused_widget != self.autocomplete_listbox and focused_widget != self.relationship_label_entry:
                self._destroy_autocomplete_listbox()
        except KeyError:
            self._destroy_autocomplete_listbox()
        except Exception as e:
            print(f"Unexpected error during autocomplete focus check: {e}")
            traceback.print_exc()
            self._destroy_autocomplete_listbox()

    def _destroy_autocomplete_listbox(self):
        if self.autocomplete_listbox and self.autocomplete_listbox.winfo_exists():
            self.autocomplete_listbox.destroy()
        self.autocomplete_listbox = None

    def load_relationship_labels(self):
        try:
            script_dir_path = os.path.dirname(os.path.abspath(__file__));
            file_path_to_load = os.path.join(
                script_dir_path, RELATIONSHIP_LABELS_FILE)
        except NameError:
            file_path_to_load = RELATIONSHIP_LABELS_FILE
        try:
            if os.path.exists(file_path_to_load):
                with open(file_path_to_load, "r") as f:
                    loaded_labels = json.load(f)
                    self.relationship_labels = set(str(lbl) for lbl in loaded_labels) if isinstance(loaded_labels,
                                                                                                    list) else set()
            else:
                self.relationship_labels = set()
        except Exception as e:
            print(f"Error loading relationship labels from {file_path_to_load}: {e}");
            self.relationship_labels = set()

    def save_relationship_labels(self):
        try:
            script_dir_path = os.path.dirname(os.path.abspath(__file__));
            file_path_to_save = os.path.join(
                script_dir_path, RELATIONSHIP_LABELS_FILE)
        except NameError:
            file_path_to_save = RELATIONSHIP_LABELS_FILE
        try:
            with open(file_path_to_save, "w") as f:
                json.dump(sorted(list(self.relationship_labels)), f, indent=2)
        except Exception as e:
            print(f"Error saving relationship labels to {file_path_to_save}: {e}")

    def push_to_undo_stack(self, action_item):
        if len(self.undo_stack) >= UNDO_REDO_MAX_SIZE: self.undo_stack.pop(0)
        self.undo_stack.append(action_item);
        self.redo_stack.clear();
        self.update_undo_redo_buttons()

    def push_to_redo_stack(self, action_item):
        if len(self.redo_stack) >= UNDO_REDO_MAX_SIZE: self.redo_stack.pop(0)
        self.redo_stack.append(action_item);
        self.update_undo_redo_buttons()

    def create_relationship_action(self, action_type_str, img_filename_str, src_id_str, rel_idx_int, lbl_str,
                                   tgt_id_str, new_lbl_str=None, new_tgt_id_str=None):
        action_dict = {"type": action_type_str, "image_filename": img_filename_str, "source_mask_id": src_id_str,
                       "relationship_index": rel_idx_int, "label": lbl_str, "target_id": tgt_id_str}
        if action_type_str == "update":
            action_dict["new_label"] = new_lbl_str
            action_dict["new_target_id"] = new_tgt_id_str
        return action_dict

    def update_undo_redo_buttons(self):
        undo_btn_state = NORMAL if self.undo_stack else DISABLED;
        redo_btn_state = NORMAL if self.redo_stack else DISABLED
        self.undo_button.config(state=undo_btn_state);
        self.redo_button.config(state=redo_btn_state)
        try:
            self.editmenu.entryconfig("Undo", state=undo_btn_state);
            self.editmenu.entryconfig("Redo", state=redo_btn_state)
        except TclError:
            pass
        except AttributeError:
            pass

    def undo(self):
        if self.undo_stack:
            action_to_undo = self.undo_stack.pop();
            is_success = self.apply_relationship_action(action_to_undo, reverse=True)
            if is_success:
                self.push_to_redo_stack(action_to_undo);
                self.display_selected_mask_details();
                self.redraw_image_with_masks();
                self.status_label.config(text=f"Undo: {action_to_undo['type']}")
                self.update_statistics()  # Update stats on undo
            else:
                print(f"Undo failed for action: {action_to_undo}");
                self.status_label.config(text="Undo failed")
                self.update_undo_redo_buttons()
        else:
            self.status_label.config(text="Nothing to undo")

    def redo(self):
        if self.redo_stack:
            action_to_redo = self.redo_stack.pop();
            is_success = self.apply_relationship_action(action_to_redo, reverse=False)
            if is_success:
                self.push_to_undo_stack(action_to_redo);
                self.display_selected_mask_details();
                self.redraw_image_with_masks();
                self.status_label.config(text=f"Redo: {action_to_redo['type']}")
                self.update_statistics()  # Update stats on redo
            else:
                print(f"Redo failed for action: {action_to_redo}");
                self.status_label.config(text="Redo failed")
                self.update_undo_redo_buttons()
        else:
            self.status_label.config(text="Nothing to redo")

    def apply_relationship_action(self, action_data, reverse_op=False):
        action_type = action_data["type"]
        img_filename = action_data["image_filename"]
        src_mask_id = action_data["source_mask_id"]
        rel_idx = action_data["relationship_index"]
        original_label = action_data["label"]
        original_target_id = action_data["target_id"]
        source_mask_obj = self.get_mask_by_id(img_filename, src_mask_id)
        if not source_mask_obj:
            print(f"Apply Action Error: Source mask ID '{src_mask_id}' not found in image '{img_filename}'.");
            return False
        relationships_list = source_mask_obj["relationships"]
        try:
            if action_type == "add":
                item_data = {"label": original_label, "target_id": original_target_id}
                if reverse_op:
                    if item_data in relationships_list:
                        relationships_list.remove(item_data)
                    else:
                        if rel_idx is not None and 0 <= rel_idx < len(relationships_list) + 1:
                            print(
                                f"Undo Add Warning: Item {item_data} not found by value. Index {rel_idx} might be stale.")
                        else:
                            print(f"Undo Add Warning: Item {item_data} not found by value and no valid index provided.")
                else:
                    relationships_list.append(item_data)
            elif action_type == "update":
                new_label_val = action_data["new_label"]
                new_target_id_val = action_data["new_target_id"]
                if not (rel_idx is not None and 0 <= rel_idx < len(relationships_list)):
                    print(f"Apply Update Error: Invalid relationship index '{rel_idx}'.");
                    return False
                if reverse_op:
                    relationships_list[rel_idx].update({"label": original_label, "target_id": original_target_id})
                else:
                    relationships_list[rel_idx].update({"label": new_label_val, "target_id": new_target_id_val})
            elif action_type == "delete":
                item_data_to_restore = {"label": original_label, "target_id": original_target_id}
                if reverse_op:
                    if rel_idx is not None:
                        relationships_list.insert(rel_idx, item_data_to_restore)
                    else:
                        relationships_list.append(item_data_to_restore);
                        print(
                            "Undo Delete Warning: No index, appended.")
                else:
                    if rel_idx is not None and 0 <= rel_idx < len(relationships_list) and \
                            relationships_list[rel_idx]["label"] == original_label and \
                            relationships_list[rel_idx]["target_id"] == original_target_id:
                        relationships_list.pop(rel_idx)
                    else:
                        if item_data_to_restore in relationships_list:
                            relationships_list.remove(item_data_to_restore)
                            print(
                                f"Redo Delete Warning: Item removed by value as index {rel_idx} was invalid or item mismatched.")
                        else:
                            print(
                                f"Redo Delete Error: Item at index {rel_idx} mismatch or item not found for deletion.");
                            return False
            return True
        except (ValueError, IndexError, KeyError) as e:
            print(f"Error applying action ({action_type}, reverse={reverse_op}): {e}");
            traceback.print_exc();
            return False

    def add_relationship(self, image_filename_str, source_mask_id_str, rel_label_str, target_id_str):
        if not image_filename_str or not source_mask_id_str:
            messagebox.showerror("Error", "Image or source mask not specified for adding relationship.")
            return False
        source_mask_obj = self.get_mask_by_id(image_filename_str, source_mask_id_str)
        if not source_mask_obj:
            messagebox.showerror("Error",
                                 f"Source mask ID '{source_mask_id_str}' not found in image '{image_filename_str}'.")
            return False
        for rel_item in source_mask_obj["relationships"]:
            if rel_item["label"] == rel_label_str and rel_item["target_id"] == target_id_str:
                messagebox.showwarning("Duplicate",
                                       f"The relationship '{rel_label_str}' to target '{target_id_str}' already exists for this feature.")
                return False
        new_relationship_dict = {"label": rel_label_str, "target_id": target_id_str}
        source_mask_obj["relationships"].append(new_relationship_dict)
        if rel_label_str not in self.relationship_labels:
            self.relationship_labels.add(rel_label_str)
        return True

    def on_add_relationship(self):
        if not self.current_image or not self.selected_mask_id:
            messagebox.showwarning("Input Required", "Select image and source feature.");
            return
        rel_label_val = self.relationship_label_var.get().strip()
        target_display_val = self.target_mask_id_var.get()
        if not rel_label_val:
            messagebox.showwarning("Input Required", "Enter relationship label.");
            self.relationship_label_entry.focus_set();
            return
        if not target_display_val or not hasattr(self, '_target_mask_map'):
            messagebox.showwarning("Input Required", "Select target feature.");
            return
        target_id_val = self._target_mask_map.get(target_display_val)
        if not target_id_val:
            messagebox.showerror("Internal Error", "Cannot map target display to ID.");
            return
        if self.add_relationship(self.current_image, self.selected_mask_id, rel_label_val, target_id_val):
            source_mask_obj = self.get_mask_by_id(self.current_image, self.selected_mask_id)
            added_rel_idx = -1
            if source_mask_obj:
                for i in range(len(source_mask_obj["relationships"]) - 1, -1, -1):
                    r_item = source_mask_obj["relationships"][i]
                    if r_item["label"] == rel_label_val and r_item["target_id"] == target_id_val:
                        added_rel_idx = i;
                        break
            if added_rel_idx != -1:
                self.push_to_undo_stack(
                    self.create_relationship_action("add", self.current_image, self.selected_mask_id,
                                                    added_rel_idx, rel_label_val, target_id_val))
            else:
                print(
                    f"Error: Could not find index for undo after adding relationship: {rel_label_val} -> {target_id_val}")
            self.display_selected_mask_details()
            self.clear_relationship_input()
            self.status_label.config(text=f"Added relationship: {rel_label_val} -> Target ID {target_id_val}")
            self.update_statistics()  # Update stats after adding

    def on_edit_relationship(self):
        if not self.current_image or not self.selected_mask_id:
            messagebox.showwarning("Input Required", "Select image and source feature.");
            return
        selected_idx_tuple = self.relationship_listbox.curselection()
        if not selected_idx_tuple:
            messagebox.showwarning("Selection Required", "Select relationship to update first.");
            return
        idx_in_list_val = selected_idx_tuple[0]
        new_label_val = self.relationship_label_var.get().strip()
        new_target_display_val = self.target_mask_id_var.get()
        if not new_label_val:
            messagebox.showwarning("Input Required", "Enter relationship label.");
            self.relationship_label_entry.focus_set();
            return
        if not new_target_display_val or not hasattr(self, '_target_mask_map'):
            messagebox.showwarning("Input Required", "Select target feature.");
            return
        new_target_id_val = self._target_mask_map.get(new_target_display_val)
        if not new_target_id_val:
            messagebox.showerror("Internal Error", "Cannot map target display to ID for update.");
            return
        if self.update_relationship(self.current_image, self.selected_mask_id, idx_in_list_val, new_label_val,
                                    new_target_id_val):
            self.display_selected_mask_details();
            self.clear_relationship_input();
            self.status_label.config(text="Relationship updated")
            self.update_statistics()  # Update stats after editing

    def on_delete_relationship(self):
        if not self.current_image or not self.selected_mask_id:
            messagebox.showwarning("Input Required", "Select image and source feature.");
            return
        selected_idx_tuple = self.relationship_listbox.curselection()
        if selected_idx_tuple:
            idx_in_list_val = selected_idx_tuple[0]
            selected_rel_text_val = self.relationship_listbox.get(idx_in_list_val)
            if messagebox.askyesno("Confirm Delete", f"Delete this relationship?\n\n{selected_rel_text_val}"):
                if self.delete_relationship(self.current_image, self.selected_mask_id, idx_in_list_val):
                    self.display_selected_mask_details();
                    self.clear_relationship_input();
                    self.status_label.config(text="Relationship deleted")
                    self.update_statistics()  # Update stats after deleting
                else:
                    messagebox.showerror("Error", "Failed to delete relationship.")
        else:
            messagebox.showwarning("Selection Required", "Select relationship to delete first.")

    def clear_relationship_input(self):
        self.relationship_label_var.set("")
        self.target_mask_id_combobox.set("")
        if self.relationship_listbox.curselection(): self.relationship_listbox.selection_clear(0, END)
        if self.selected_target_mask_id is not None:
            self.selected_target_mask_id = None;
            self.redraw_image_with_masks()
        self.update_button_states()

    def update_relationship(self, image_filename_str, source_mask_id_str, rel_idx_in_list_int, new_label_str,
                            new_target_id_str):
        source_mask_obj = self.get_mask_by_id(image_filename_str, source_mask_id_str)
        if not source_mask_obj:
            messagebox.showerror("Error", f"Source mask ID '{source_mask_id_str}' not found.");
            return False
        if not (0 <= rel_idx_in_list_int < len(source_mask_obj["relationships"])):
            messagebox.showerror("Error", "Invalid relationship index for update.");
            return False
        original_relationship_dict = source_mask_obj["relationships"][rel_idx_in_list_int].copy()
        for i, rel_item in enumerate(source_mask_obj["relationships"]):
            if i == rel_idx_in_list_int: continue
            if rel_item["label"] == new_label_str and rel_item["target_id"] == new_target_id_str:
                messagebox.showwarning("Duplicate",
                                       f"Another relationship with label '{new_label_str}' to target '{new_target_id_str}' already exists.")
                return False
        source_mask_obj["relationships"][rel_idx_in_list_int]["label"] = new_label_str
        source_mask_obj["relationships"][rel_idx_in_list_int]["target_id"] = new_target_id_str
        if new_label_str not in self.relationship_labels: self.relationship_labels.add(new_label_str)
        self.push_to_undo_stack(
            self.create_relationship_action("update", image_filename_str, source_mask_id_str, rel_idx_in_list_int,
                                            original_relationship_dict["label"],
                                            original_relationship_dict["target_id"],
                                            new_label_str, new_target_id_str))
        return True

    def delete_relationship(self, image_filename_str, source_mask_id_str, rel_idx_in_list_int):
        source_mask_obj = self.get_mask_by_id(image_filename_str, source_mask_id_str)
        if not source_mask_obj:
            messagebox.showerror("Error", f"Source mask ID '{source_mask_id_str}' not found.");
            return False
        if not (0 <= rel_idx_in_list_int < len(source_mask_obj["relationships"])):
            messagebox.showerror("Error", "Invalid relationship index for delete.");
            return False
        deleted_relationship_dict = source_mask_obj["relationships"].pop(rel_idx_in_list_int)
        self.push_to_undo_stack(
            self.create_relationship_action("delete", image_filename_str, source_mask_id_str,
                                            rel_idx_in_list_int,
                                            deleted_relationship_dict["label"], deleted_relationship_dict["target_id"]))
        return True


def main():
    root = Tk()
    root.minsize(900, 700)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    app = SceneGraphAnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()