import customtkinter as ctk
import xml.etree.ElementTree as ET
from tkinter import filedialog, messagebox, Menu
from PIL import Image, ImageTk
import json
import os
import cv2
import numpy as np
import traceback
from collections import Counter, defaultdict
import webbrowser
from create_graph import create_knowledge_graph

RELATIONSHIP_LABELS_FILE = "v6/relationship_labels.json"
UNDO_REDO_MAX_SIZE = 20

# Define colors for masks and bounding boxes (BGR format for OpenCV)
# Using CTk theme colors can be an alternative, but for masks, distinct colors are better.
SOURCE_MASK_COLOR = (0, 0, 255)  # Red
SOURCE_BBOX_COLOR = (0, 255, 0)  # Green
TARGET_MASK_COLOR = (255, 0, 0)  # Blue
TARGET_BBOX_COLOR = (0, 255, 255)  # Yellow


class SceneGraphAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Modern Scene Graph Annotation Tool")
        self.root.geometry("1200x800")  # Set a default size

        # --- CustomTkinter Settings ---
        ctk.set_appearance_mode("Dark")  # Options: "System", "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

        # --- Data Structures ---
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
        self.stats_window = None

        # To keep track of listbox buttons for selection highlighting
        self.image_list_buttons = []
        self.feature_list_buttons = []
        self.relationship_list_buttons = []

        # --- GUI Elements ---
        self.create_widgets()
        self.load_relationship_labels()

    def create_widgets(self):
        """Creates and arranges all the GUI widgets using CustomTkinter."""

        # --- Main Layout (3 Columns) ---
        self.root.grid_columnconfigure(0, minsize=420, weight=0)  # Left sidebar
        self.root.grid_columnconfigure(1, weight=3)  # Center image area
        self.root.grid_columnconfigure(2, minsize=420, weight=0)  # Right sidebar
        self.root.grid_rowconfigure(0, weight=1)

        # --- Left Sidebar (Scrollable) ---
        self.sidebar_scroll_frame = ctk.CTkScrollableFrame(self.root, width=420, corner_radius=0)
        self.sidebar_scroll_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar_scroll_frame.grid_columnconfigure(0, weight=1)
        self.sidebar_scroll_frame.grid_rowconfigure(4, weight=1)  # Let image list expand

        # --- Center Main Area (Image) ---
        self.image_frame = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        self.image_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.image_frame.grid_rowconfigure(1, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)

        # --- Right Sidebar (Scrollable) ---
        self.right_sidebar_scroll_frame = ctk.CTkScrollableFrame(self.root, width=420, corner_radius=0)
        self.right_sidebar_scroll_frame.grid(row=0, column=2, rowspan=2, sticky="nsew")
        self.right_sidebar_scroll_frame.grid_columnconfigure(0, weight=1)
        # Let the list frames in the right sidebar expand
        self.right_sidebar_scroll_frame.grid_rowconfigure(2, minsize=150, weight=1)
        self.right_sidebar_scroll_frame.grid_rowconfigure(5, minsize=100, weight=1)
        self.right_sidebar_scroll_frame.grid_rowconfigure(7, minsize=100, weight=1)

        # ===================================================================
        # WIDGETS FOR THE LEFT SIDEBAR
        # ===================================================================
        self.logo_label = ctk.CTkLabel(self.sidebar_scroll_frame, text="Annotation Tool",
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.file_button_frame = ctk.CTkFrame(self.sidebar_scroll_frame, fg_color="transparent")
        self.file_button_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.file_button_frame.grid_columnconfigure((0, 1), weight=1)
        self.load_xml_button = ctk.CTkButton(self.file_button_frame, text="Load XML", command=self.load_xml)
        self.load_xml_button.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.load_images_button = ctk.CTkButton(self.file_button_frame, text="Load Images", command=self.load_images)
        self.load_images_button.grid(row=0, column=1, padx=(5, 0), sticky="ew")

        self.save_button = ctk.CTkButton(self.sidebar_scroll_frame, text="Save Annotations", command=self.save_xml)
        self.save_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")


        # --- Image List (formerly in a tab) ---
        self.image_list_label = ctk.CTkLabel(self.sidebar_scroll_frame, text="Image Files", anchor="w")
        self.image_list_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="ew")
        self.image_list_scrollable_frame = ctk.CTkScrollableFrame(self.sidebar_scroll_frame, label_text="")
        self.image_list_scrollable_frame.grid(row=4, column=0, padx=20, pady=5, sticky="nsew")

        # --- Statistics (formerly in a tab) ---
        self.stats_label = ctk.CTkLabel(self.sidebar_scroll_frame, text="Statistics",
                                        font=ctk.CTkFont(size=14, weight="bold"))
        self.stats_label.grid(row=5, column=0, padx=20, pady=(15, 0), sticky="ew")
        self.stats_container_frame = ctk.CTkFrame(self.sidebar_scroll_frame, fg_color="transparent")
        self.stats_container_frame.grid(row=6, column=0, padx=20, pady=10, sticky="ew")
        self.stats_container_frame.grid_columnconfigure(1, weight=1)

        self.total_images_var = ctk.StringVar(value="0")
        self.total_features_var = ctk.StringVar(value="0")
        self.total_attributes_var = ctk.StringVar(value="0")
        self.total_relationships_var = ctk.StringVar(value="0")

        ctk.CTkLabel(self.stats_container_frame, text="Images:", anchor="w").grid(row=0, column=0, sticky="w", pady=2)
        ctk.CTkLabel(self.stats_container_frame, textvariable=self.total_images_var, anchor="e").grid(row=0, column=1,
                                                                                                      sticky="e",
                                                                                                      pady=2)
        ctk.CTkLabel(self.stats_container_frame, text="Features:", anchor="w").grid(row=1, column=0, sticky="w", pady=2)
        ctk.CTkLabel(self.stats_container_frame, textvariable=self.total_features_var, anchor="e").grid(row=1, column=1,
                                                                                                        sticky="e",
                                                                                                        pady=2)
        ctk.CTkLabel(self.stats_container_frame, text="Attributes:", anchor="w").grid(row=2, column=0, sticky="w",
                                                                                      pady=2)
        ctk.CTkLabel(self.stats_container_frame, textvariable=self.total_attributes_var, anchor="e").grid(row=2,
                                                                                                          column=1,
                                                                                                          sticky="e",
                                                                                                          pady=2)
        ctk.CTkLabel(self.stats_container_frame, text="Relationships:", anchor="w").grid(row=3, column=0, sticky="w",
                                                                                         pady=2)
        ctk.CTkLabel(self.stats_container_frame, textvariable=self.total_relationships_var, anchor="e").grid(row=3,
                                                                                                             column=1,
                                                                                                             sticky="e",
                                                                                                             pady=2)

        self.more_stats_button = ctk.CTkButton(self.stats_container_frame, text="Show Detailed Statistics",
                                               command=self.show_more_stats)
        self.more_stats_button.grid(row=4, column=0, columnspan=2, pady=(10, 0), sticky="ew")

        # ===================================================================
        # WIDGETS FOR THE RIGHT SIDEBAR
        # ===================================================================

        # --- Annotation Details (formerly in a tab) ---
        self.feature_list_label = ctk.CTkLabel(self.right_sidebar_scroll_frame, text="Features (Source)", anchor="w")
        self.feature_list_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="ew")
        self.feature_list_scrollable_frame = ctk.CTkScrollableFrame(self.right_sidebar_scroll_frame, label_text="")
        self.feature_list_scrollable_frame.grid(row=2, column=0, padx=20, pady=5, sticky="nsew")

        self.selected_feature_label = ctk.CTkLabel(self.right_sidebar_scroll_frame, text="No Feature Selected",
                                                   font=ctk.CTkFont(weight="bold"), anchor="w")
        self.selected_feature_label.grid(row=3, column=0, padx=20, pady=(10, 5), sticky="ew")

        self.detail_label = ctk.CTkLabel(self.right_sidebar_scroll_frame, text="Attributes:", anchor="w")
        self.detail_label.grid(row=4, column=0, padx=20, pady=(5, 0), sticky="ew")
        self.attribute_text = ctk.CTkTextbox(self.right_sidebar_scroll_frame, state="disabled", wrap="word")
        self.attribute_text.grid(row=5, column=0, padx=20, pady=5, sticky="nsew")

        self.relationship_label_display = ctk.CTkLabel(self.right_sidebar_scroll_frame,
                                                       text="Relationships (Click to Edit):", anchor="w")
        self.relationship_label_display.grid(row=6, column=0, padx=20, pady=(5, 0), sticky="ew")
        self.relationship_scrollable_frame = ctk.CTkScrollableFrame(self.right_sidebar_scroll_frame, label_text="")
        self.relationship_scrollable_frame.grid(row=7, column=0, padx=20, pady=5, sticky="nsew")

        # --- Relationship Input Frame ---
        self.input_frame = ctk.CTkFrame(self.right_sidebar_scroll_frame)
        self.input_frame.grid(row=8, column=0, padx=20, pady=10, sticky="ew")
        self.input_frame.grid_columnconfigure(1, weight=1)

        self.input_label = ctk.CTkLabel(self.input_frame, text="Add/Edit Relationship", font=ctk.CTkFont(weight="bold"))
        self.input_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")

        self.relationship_label_entry = ctk.CTkEntry(self.input_frame, placeholder_text="Relationship Label")
        self.relationship_label_entry.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        self.target_mask_id_combobox = ctk.CTkComboBox(self.input_frame, values=[], state="readonly",
                                                       command=self.on_target_mask_select)
        self.target_mask_id_combobox.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.target_mask_id_combobox.set("Select Target Feature")
        self.target_mask_id_combobox.bind("<Button-1>", self._open_combobox)

        self.rel_button_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        self.rel_button_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.rel_button_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.add_relationship_button = ctk.CTkButton(self.rel_button_frame, text="Add",
                                                     command=self.on_add_relationship)
        self.add_relationship_button.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.edit_relationship_button = ctk.CTkButton(self.rel_button_frame, text="Update",
                                                      command=self.on_edit_relationship)
        self.edit_relationship_button.grid(row=0, column=1, padx=5, sticky="ew")
        self.delete_relationship_button = ctk.CTkButton(self.rel_button_frame, text="Delete",
                                                        command=self.on_delete_relationship, fg_color="#D32F2F",
                                                        hover_color="#B71C1C")
        self.delete_relationship_button.grid(row=0, column=2, padx=(5, 0), sticky="ew")

        # ===================================================================
        # WIDGETS FOR THE CENTER AREA AND STATUS BAR
        # ===================================================================
        # --- Image Display Area ---
        self.image_nav_frame = ctk.CTkFrame(self.image_frame, fg_color="transparent")
        self.image_nav_frame.grid(row=0, column=0, padx=10, pady=(5, 5), sticky="ew")
        self.image_nav_frame.grid_columnconfigure(1, weight=1)

        self.image_counter_label = ctk.CTkLabel(self.image_nav_frame, text="Image 0/0", font=ctk.CTkFont(size=14))
        self.image_counter_label.grid(row=0, column=1, sticky="w")

        self.view_graph_button = ctk.CTkButton(self.image_nav_frame, text="View Graph",
                                               command=self.view_knowledge_graph, width=100)
        self.view_graph_button.grid(row=0, column=2, sticky="e", padx=(0, 5))


        self.prev_image_button = ctk.CTkButton(self.image_nav_frame, text="<", command=self.go_to_previous_image,
                                               width=40)
        self.prev_image_button.grid(row=0, column=3, sticky="e", padx=(0, 5))
        self.next_image_button = ctk.CTkButton(self.image_nav_frame, text=">", command=self.go_to_next_image, width=40)
        self.next_image_button.grid(row=0, column=4, sticky="e")

        self.image_label = ctk.CTkLabel(self.image_frame, text="Load XML and an Image Directory to begin.",
                                        corner_radius=10)
        self.image_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.image_frame.bind("<Configure>", self.on_frame_resize)

        # --- Status Bar ---
        self.status_bar_frame = ctk.CTkFrame(self.root, height=30, corner_radius=0)
        self.status_bar_frame.grid(row=1, column=1, sticky="sew", padx=10, pady=(0, 10))
        self.status_bar_frame.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(self.status_bar_frame, text="Ready", anchor="w")
        self.status_label.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        # --- Menu and Shortcuts ---
        self.menubar = Menu(self.root)
        self.filemenu = Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Open XML", command=self.load_xml, accelerator="Ctrl+O")
        self.filemenu.add_command(label="Load Images", command=self.load_images, accelerator="Ctrl+L")
        self.filemenu.add_command(label="Save XML", command=self.save_xml, accelerator="Ctrl+S")
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.root.quit)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

        self.editmenu = Menu(self.menubar, tearoff=0)
        self.editmenu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z", state="disabled")
        self.editmenu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y", state="disabled")
        self.menubar.add_cascade(label="Edit", menu=self.editmenu)
        self.root.config(menu=self.menubar)

        self.root.bind_all("<Control-o>", lambda event: self.load_xml())
        self.root.bind_all("<Control-l>", lambda event: self.load_images())
        self.root.bind_all("<Control-s>", lambda event: self.save_xml())
        self.root.bind_all("<Control-z>", lambda event: self.undo())
        self.root.bind_all("<Control-y>", lambda event: self.redo())

        self.update_button_states()
        self.update_statistics()

    # ===================================================================
    # NEW/MODIFIED WIDGET HELPER METHODS
    # ===================================================================

    def go_to_next_image(self):
        """Selects the next image in the list."""
        if not self.image_list_buttons: return

        current_index = -1
        for i, btn in enumerate(self.image_list_buttons):
            if btn.cget("fg_color") != "transparent":
                current_index = i
                break

        next_index = (current_index + 1) % len(self.image_list_buttons)
        next_image_text = self.image_list_buttons[next_index].cget("text")
        self.on_list_item_select(next_image_text, "image")

    def go_to_previous_image(self):
        """Selects the previous image in the list."""
        if not self.image_list_buttons: return

        current_index = 0
        for i, btn in enumerate(self.image_list_buttons):
            if btn.cget("fg_color") != "transparent":
                current_index = i
                break

        prev_index = (current_index - 1 + len(self.image_list_buttons)) % len(self.image_list_buttons)
        prev_image_text = self.image_list_buttons[prev_index].cget("text")
        self.on_list_item_select(prev_image_text, "image")

    def _open_combobox(self, event=None):
        """Programmatically opens the combobox dropdown menu."""
        if self.target_mask_id_combobox.state == "readonly":
            self.target_mask_id_combobox._open_dropdown_menu()

    def _clear_scrollable_frame(self, frame):
        """Helper to remove all widgets from a scrollable frame."""
        for widget in frame.winfo_children():
            widget.destroy()

    def _populate_list(self, frame, button_list, items, item_type):
        """Generic function to populate a scrollable frame with buttons."""
        self._clear_scrollable_frame(frame)
        button_list.clear()
        for item in items:
            btn = ctk.CTkButton(
                frame,
                text=item,
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray70", "gray30"),
                anchor="w",
                command=lambda i=item, t=item_type: self.on_list_item_select(i, t)
            )
            btn.pack(fill="x", padx=5, pady=2)
            button_list.append(btn)

    def on_list_item_select(self, selected_item, item_type):
        """Handles selection for all custom listboxes."""
        if item_type == "image":
            self.on_image_select(selected_item)
            self._update_list_selection(self.image_list_buttons, selected_item)
        elif item_type == "feature":
            self.on_mask_select(selected_item)
            self._update_list_selection(self.feature_list_buttons, selected_item)
        elif item_type == "relationship":
            self.on_relationship_list_select(selected_item)
            self._update_list_selection(self.relationship_list_buttons, selected_item)

    def _update_list_selection(self, button_list, selected_item_text):
        """Highlights the selected button in a custom listbox."""
        for btn in button_list:
            if btn.cget("text") == selected_item_text:
                btn.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
            else:
                btn.configure(fg_color="transparent")

    # ===================================================================
    # STATISTICS WINDOW
    # ===================================================================

    def show_more_stats(self):
        """Creates and shows the detailed statistics window."""
        if not self.image_data:
            messagebox.showwarning("No Data", "Please load an XML file before viewing detailed statistics.")
            return

        if self.stats_window is None or not self.stats_window.winfo_exists():
            self.stats_window = ctk.CTkToplevel(self.root)
            self.stats_window.title("Detailed Statistics")
            self.stats_window.geometry("600x500")
            self.stats_window.transient(self.root)

            stats_tab_view = ctk.CTkTabview(self.stats_window)
            stats_tab_view.pack(expand=True, fill="both", padx=10, pady=10)
            stats_tab_view.add("Features")
            stats_tab_view.add("Attributes")
            stats_tab_view.add("Relationships")

            self.populate_feature_stats(stats_tab_view.tab("Features"))
            self.populate_attribute_stats(stats_tab_view.tab("Attributes"))
            self.populate_relationship_stats(stats_tab_view.tab("Relationships"))
        else:
            self.stats_window.focus()

    def _populate_stats_tab(self, parent_tab, content_generator):
        """Helper to create a textbox and populate it with formatted stats data."""
        textbox = ctk.CTkTextbox(parent_tab, wrap="word", state="disabled")
        textbox.pack(expand=True, fill="both", padx=5, pady=5)

        # Define text styles (tags) for indentation
        textbox.tag_config("title", foreground="#5DADE2")
        textbox.tag_config("item", lmargin1=20)      # First level indent
        textbox.tag_config("sub_item", lmargin1=40)  # Second level indent

        textbox.configure(state="normal")
        content_generator(textbox)  # Let the specific function insert content
        textbox.configure(state="disabled")

    def populate_feature_stats(self, tab):
        def generator(textbox):
            feature_counts = Counter()
            for img_data in self.image_data.values():
                for mask in img_data.get("masks", []):
                    feature_counts[mask.get("label", "N/A")] += 1

            textbox.insert("end", "Feature Label Frequencies\n\n", "title")
            if not feature_counts:
                textbox.insert("end", "No data available.", "item")
                return
            for key, value in sorted(feature_counts.items(), key=lambda item: item[1], reverse=True):
                textbox.insert("end", f"{key}: {value}\n", "item")

        self._populate_stats_tab(tab, generator)

    def populate_attribute_stats(self, tab):
        def generator(textbox):
            # Use a nested dictionary for the new format: {feature: {attribute: {option: count}}}
            attr_stats = defaultdict(lambda: defaultdict(Counter))

            # Gather the statistics from your image data
            for img_data in self.image_data.values():
                for mask in img_data.get("masks", []):
                    feature_label = mask.get("label", "N/A")
                    for attr_name, attr_value in mask.get("attributes", {}).items():
                        attr_stats[feature_label][attr_name][str(attr_value)] += 1

            if not attr_stats:
                textbox.insert("end", "No data available.", "item")
                return

            # Populate the textbox with the new, nested format
            for feature, attributes in sorted(attr_stats.items()):
                textbox.insert("end", f"{feature}\n", "title") # <Feature>
                for attr_name, options in sorted(attributes.items()):
                    textbox.insert("end", f"{attr_name}:\n", "item") # <Attribute>:
                    # Sort options by count (descending), then alphabetically for ties
                    for option_value, count in sorted(options.items(), key=lambda item: (-item[1], item[0])):
                        textbox.insert("end", f"{option_value}: {count}\n", "sub_item") # <Option>: <Count>
                textbox.insert("end", "\n")

        self._populate_stats_tab(tab, generator)

    def populate_relationship_stats(self, tab):
        def generator(textbox):
            overall_rel_counts = Counter()
            rel_by_feature = defaultdict(Counter)

            for img_data in self.image_data.values():
                for mask in img_data.get("masks", []):
                    feature_label = mask.get("label", "N/A")
                    for rel in mask.get("relationships", []):
                        rel_label = rel.get("label", "N/A")
                        overall_rel_counts[rel_label] += 1
                        rel_by_feature[feature_label][rel_label] += 1

            textbox.insert("end", "Overall Relationship Frequencies\n\n", "title")
            if not overall_rel_counts:
                textbox.insert("end", "No data available.\n", "item")
            else:
                for rel, count in sorted(overall_rel_counts.items(), key=lambda item: item[1], reverse=True):
                    textbox.insert("end", f"{rel}: {count}\n", "item")

            textbox.insert("end", "\n\n")

            textbox.insert("end", "Relationships by Source Feature\n\n", "title")
            if not rel_by_feature:
                textbox.insert("end", "No data available.\n", "item")
            else:
                for feature, counts in sorted(rel_by_feature.items()):
                    textbox.insert("end", f"Feature: '{feature}'\n", "title")
                    for rel, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
                        textbox.insert("end", f"{rel}: {count}\n", "item")
                    textbox.insert("end", "\n")

        self._populate_stats_tab(tab, generator)

    # ===================================================================
    # CORE APPLICATION LOGIC (Mostly unchanged, but adapted for CTk)
    # ===================================================================

    def update_statistics(self):
        """Calculates and updates the statistics display panel."""
        if not self.image_data:
            self.total_images_var.set("0")
            self.total_features_var.set("0")
            self.total_attributes_var.set("0")
            self.total_relationships_var.set("0")
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

        self.total_images_var.set(f"{num_images}")
        self.total_features_var.set(f"{num_features}")
        self.total_attributes_var.set(f"{num_attributes}")
        self.total_relationships_var.set(f"{num_relationships}")

    def update_button_states(self):
        is_rel_selected = any(btn.cget("fg_color") != "transparent" for btn in self.relationship_list_buttons)

        if is_rel_selected:
            self.add_relationship_button.configure(state="disabled")
            self.edit_relationship_button.configure(state="normal")
            self.delete_relationship_button.configure(state="normal")
        else:
            self.add_relationship_button.configure(state="normal")
            self.edit_relationship_button.configure(state="disabled")
            self.delete_relationship_button.configure(state="disabled")

    def on_frame_resize(self, event=None):
        if self._resize_timer:
            self.root.after_cancel(self._resize_timer)
        self._resize_timer = self.root.after(100, self.redraw_image_with_masks)

    def load_xml(self):
        file_path = filedialog.askopenfilename(filetypes=[("XML files", "*.xml")])
        if file_path:
            try:
                self.status_label.configure(text="Loading XML...")
                self.root.update_idletasks()

                self.loaded_xml_path = file_path
                self.image_data = self.load_cvat_xml(file_path)
                self.populate_image_list()

                if self.image_list_buttons:
                    first_image_text = self.image_list_buttons[0].cget("text")
                    self.on_list_item_select(first_image_text, "image")

                self.status_label.configure(text=f"Loaded: {os.path.basename(file_path)}")
                self.undo_stack = []
                self.redo_stack = []
                self.update_undo_redo_buttons()
                self.update_statistics()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load XML: {e}\n\nCheck XML format and file permissions.")
                self.status_label.configure(text="Error loading XML")
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
                    self.status_label.configure(text=f"Image directory: {self.image_directory}")
                else:
                    messagebox.showwarning("Image Not Found",
                                           f"Current image '{self.current_image}' not found in the selected directory.")
                    self.image_label.configure(image=None, text=f"Image not found:\n{self.current_image}")
                    self.tk_image = None
                    self.status_label.configure(text=f"Image dir set, but '{self.current_image}' not found.")
            else:
                self.status_label.configure(text=f"Image directory set. Select an image to view.")

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
                self.status_label.configure(text="Saving...")
                self.root.update_idletasks()
                self.update_xml_with_relationships(self.image_data, self.loaded_xml_path, file_path)
                self.save_relationship_labels()
                self.status_label.configure(text=f"Saved XML to: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Annotations saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save XML: {e}")
                traceback.print_exc()
                self.status_label.configure(text="Error saving XML")

    def populate_image_list(self):
        items = [f"{i + 1}) {filename}" for i, filename in enumerate(sorted(self.image_data.keys()))]
        self._populate_list(self.image_list_scrollable_frame, self.image_list_buttons, items, "image")

    def populate_feature_list(self):
        items = []
        if self.current_image and self.current_image in self.image_data:
            try:
                sorted_masks = sorted(self.image_data[self.current_image]["masks"], key=lambda m: int(m['id']))
            except (ValueError, TypeError):
                sorted_masks = sorted(self.image_data[self.current_image]["masks"], key=lambda m: m['id'])
            items = [f"{m['id']}:{m['label']}" for m in sorted_masks]

        self._populate_list(self.feature_list_scrollable_frame, self.feature_list_buttons, items, "feature")
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

        self.target_mask_id_combobox.configure(values=current_values)
        current_selection = self.target_mask_id_combobox.get()
        if current_selection not in self._target_mask_map:
            self.target_mask_id_combobox.set('Select Target Feature')
            self.selected_target_mask_id = None

    def redraw_image_with_masks(self):
        if not self.current_image or not self.image_directory:
            self.image_label.configure(image=None, text="Select an image to display")
            self.tk_image = None
            return
        image_path = os.path.join(self.image_directory, self.current_image)
        if not os.path.exists(image_path):
            self.image_label.configure(image=None, text=f"Image not found:\n{self.current_image}")
            self.tk_image = None
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

    def draw_masks_on_image(self, image_path, source_masks, target_masks=None):
        if target_masks is None: target_masks = []
        try:
            frame_w = self.image_label.winfo_width()
            frame_h = self.image_label.winfo_height()
            if frame_w <= 1 or frame_h <= 1: return

            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                self.image_label.configure(image=None, text=f"Error reading image:\n{os.path.basename(image_path)}")
                return

            orig_h, orig_w, _ = img_bgr.shape
            aspect_ratio = orig_w / orig_h
            disp_w, disp_h = frame_w, int(frame_w / aspect_ratio)
            if disp_h > frame_h:
                disp_h = frame_h
                disp_w = int(disp_h * aspect_ratio)

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
                                y_start, y_end = max(0, top_i), min(orig_h, top_i + bbox_h_i)
                                x_start, x_end = max(0, left_i), min(orig_w, left_i + bbox_w_i)
                                y_slice, x_slice = slice(y_start - top_i, y_end - top_i), slice(x_start - left_i,
                                                                                                x_end - left_i)
                                if y_end > y_start and x_end > x_start:
                                    try:
                                        img_slice = full_mask_original_res[y_start:y_end, x_start:x_end]
                                        mask_slice = decoded_bbox_mask[y_slice, x_slice]
                                        if img_slice.shape == mask_slice.shape:
                                            img_slice[:] = mask_slice
                                    except Exception:
                                        pass
                                full_mask_display_res = cv2.resize(full_mask_original_res, (disp_w, disp_h),
                                                                   interpolation=cv2.INTER_NEAREST)
                                color_overlay[full_mask_display_res == 1] = mask_color_val
                    if bbox:
                        try:
                            left_f_bbox, top_f_bbox, width_f_bbox, height_f_bbox = map(float, bbox)
                        except (ValueError, TypeError):
                            continue
                        if width_f_bbox > 0 and height_f_bbox > 0:
                            scale_x, scale_y = disp_w / orig_w, disp_h / orig_h
                            l, t = int(left_f_bbox * scale_x), int(top_f_bbox * scale_y)
                            r, b = int((left_f_bbox + width_f_bbox) * scale_x), int(
                                (top_f_bbox + height_f_bbox) * scale_y)
                            cv2.rectangle(img_display_bgr, (l, t), (r, b), bbox_draw_color_val, 2)

            alpha_blend = 0.4
            active_mask_indices = np.any(color_overlay > 0, axis=-1)
            if np.any(active_mask_indices):
                src1 = color_overlay[active_mask_indices].astype(img_display_bgr.dtype)
                src2 = img_display_bgr[active_mask_indices].astype(img_display_bgr.dtype)
                img_display_bgr[active_mask_indices] = cv2.addWeighted(src1, alpha_blend, src2, 1.0 - alpha_blend, 0)

            img_rgb_final = cv2.cvtColor(img_display_bgr, cv2.COLOR_BGR2RGB)
            img_pil_final = Image.fromarray(img_rgb_final)

            self.tk_image = ctk.CTkImage(light_image=img_pil_final, dark_image=img_pil_final, size=(disp_w, disp_h))
            self.image_label.configure(image=self.tk_image, text="")

        except Exception as e:
            print(f"--- Error during mask/image drawing for {self.current_image} ---")
            traceback.print_exc()
            self.image_label.configure(image=None, text="Error drawing image")
            self.tk_image = None

    def display_selected_mask_details(self):
        self.attribute_text.configure(state="normal")
        self.attribute_text.delete("1.0", "end")
        self._clear_scrollable_frame(self.relationship_scrollable_frame)
        self.relationship_list_buttons.clear()

        mask_details_data = self.get_mask_by_id(self.current_image,
                                                self.selected_mask_id) if self.current_image and self.selected_mask_id else None

        if mask_details_data:
            self.selected_feature_label.configure(
                text=f"Selected: {mask_details_data['id']}:{mask_details_data['label']}")
            attrs_str = "\n".join(f"{k}: {v}" for k, v in sorted(mask_details_data["attributes"].items()))
            self.attribute_text.insert("end", attrs_str if attrs_str else "No attributes")

            if mask_details_data["relationships"]:
                current_img_masks = self.image_data.get(self.current_image, {}).get("masks", [])
                id_to_label_map = {m['id']: m['label'] for m in current_img_masks}
                try:
                    sorted_rels_list = sorted(mask_details_data["relationships"],
                                              key=lambda r: (r['label'], int(r['target_id'])))
                except (ValueError, TypeError):
                    sorted_rels_list = sorted(mask_details_data["relationships"],
                                              key=lambda r: (r['label'], r['target_id']))

                rel_items = [f"{r['label']} -> {r['target_id']}:{id_to_label_map.get(r['target_id'], 'Unknown')}" for r
                             in sorted_rels_list]
                self._populate_list(self.relationship_scrollable_frame, self.relationship_list_buttons, rel_items,
                                    "relationship")

            else:
                no_rel_label = ctk.CTkLabel(self.relationship_scrollable_frame, text="(No relationships)",
                                            text_color="gray")
                no_rel_label.pack(padx=5, pady=5)
        else:
            self.selected_feature_label.configure(text="No Feature Selected")
            self.attribute_text.insert("end", "N/A")
            no_rel_label = ctk.CTkLabel(self.relationship_scrollable_frame, text="N/A", text_color="gray")
            no_rel_label.pack(padx=5, pady=5)

        self.attribute_text.configure(state="disabled")
        self.clear_relationship_input()
        self.update_statistics()

    def clear_all_lists_and_details(self):
        self._clear_scrollable_frame(self.image_list_scrollable_frame)
        self._clear_scrollable_frame(self.feature_list_scrollable_frame)
        self._clear_scrollable_frame(self.relationship_scrollable_frame)
        self.image_list_buttons.clear()
        self.feature_list_buttons.clear()
        self.relationship_list_buttons.clear()

        self.attribute_text.configure(state="normal")
        self.attribute_text.delete('1.0', 'end')
        self.attribute_text.configure(state="disabled")

        self.selected_feature_label.configure(text="No Feature Selected")
        self.clear_relationship_input()
        self.image_label.configure(image=None, text="Load XML and an Image Directory to begin.")
        self.tk_image = None
        self.current_image = None
        self.selected_mask_id = None
        self.image_counter_label.configure(text="Image 0/0")
        self.update_statistics()

    def on_image_select(self, selected_text):
        if not selected_text: return
        try:
            idx_str, new_img_filename = selected_text.split(") ", 1)
            idx = int(idx_str) - 1
        except ValueError:
            return

        total_images = len(self.image_list_buttons)
        self.image_counter_label.configure(text=f"Image {idx + 1}/{total_images}")

        if new_img_filename != self.current_image:
            self.current_image = new_img_filename
            self.populate_feature_list()
            self.selected_mask_id = None
            self.clear_relationship_input()
            self.populate_feature_dropdown()
            self.display_selected_mask_details()
            self.redraw_image_with_masks()

    def on_mask_select(self, selected_text):
        if not selected_text: return
        try:
            new_selected_id = selected_text.split(":", 1)[0]
        except (ValueError, IndexError):
            return

        if new_selected_id != self.selected_mask_id:
            self.selected_mask_id = new_selected_id
            self.clear_relationship_input()
            self.populate_feature_dropdown()
            self.display_selected_mask_details()
            self.redraw_image_with_masks()

    def on_target_mask_select(self, selected_text=None):
        if not selected_text or selected_text == "Select Target Feature":
            self.selected_target_mask_id = None
            self.redraw_image_with_masks()
            return

        new_target_id_val = self._target_mask_map.get(selected_text)
        if new_target_id_val != self.selected_target_mask_id:
            self.selected_target_mask_id = new_target_id_val
            self.redraw_image_with_masks()

    def on_relationship_list_select(self, selected_rel_text):
        if not selected_rel_text:
            self.update_button_states()
            return
        try:
            label_text_part, target_text_part = selected_rel_text.split(" -> ", 1)
            target_id_str, target_label_from_list = target_text_part.split(":", 1)
            target_display_text = f"{target_id_str}:{target_label_from_list.strip()}"

            self.relationship_label_entry.delete(0, "end")
            self.relationship_label_entry.insert(0, label_text_part.strip())

            if target_display_text in self._target_mask_map:
                self.target_mask_id_combobox.set(target_display_text)
            else:
                found = False
                for display_val, t_id_val in self._target_mask_map.items():
                    if t_id_val == target_id_str:
                        self.target_mask_id_combobox.set(display_val)
                        found = True
                        break
                if not found: self.target_mask_id_combobox.set("Select Target Feature")

            self.on_target_mask_select(self.target_mask_id_combobox.get())
        except ValueError:
            print(f"Error parsing relationship selection: {selected_rel_text}")
        finally:
            self.update_button_states()

    def on_add_relationship(self):
        if not self.current_image or not self.selected_mask_id:
            messagebox.showwarning("Input Required", "Select an image and a source feature first.")
            return
        rel_label_val = self.relationship_label_entry.get().strip()
        target_display_val = self.target_mask_id_combobox.get()

        if not rel_label_val:
            messagebox.showwarning("Input Required", "Please enter a relationship label.")
            return
        if not target_display_val or target_display_val == "Select Target Feature":
            messagebox.showwarning("Input Required", "Please select a target feature.")
            return

        target_id_val = self._target_mask_map.get(target_display_val)
        if not target_id_val:
            messagebox.showerror("Internal Error", "Cannot map target display to an ID.")
            return

        if self.add_relationship(self.current_image, self.selected_mask_id, rel_label_val, target_id_val):
            source_mask_obj = self.get_mask_by_id(self.current_image, self.selected_mask_id)
            added_rel_idx = -1
            if source_mask_obj:
                for i, r_item in enumerate(source_mask_obj["relationships"]):
                    if r_item["label"] == rel_label_val and r_item["target_id"] == target_id_val:
                        added_rel_idx = i
                        break

            if added_rel_idx != -1:
                self.push_to_undo_stack(
                    self.create_relationship_action("add", self.current_image, self.selected_mask_id,
                                                    added_rel_idx, rel_label_val, target_id_val))

            self.display_selected_mask_details()
            self.clear_relationship_input()
            self.status_label.configure(text=f"Added: {rel_label_val} -> Target ID {target_id_val}")
            self.update_statistics()

    def on_edit_relationship(self):
        is_rel_selected = any(btn.cget("fg_color") != "transparent" for btn in self.relationship_list_buttons)
        if not is_rel_selected:
            messagebox.showwarning("Selection Required", "Select a relationship from the list to update.")
            return

        selected_rel_text = ""
        for btn in self.relationship_list_buttons:
            if btn.cget("fg_color") != "transparent":
                selected_rel_text = btn.cget("text")
                break

        source_mask_obj = self.get_mask_by_id(self.current_image, self.selected_mask_id)
        if not source_mask_obj: return

        idx_in_data = -1
        try:
            label_part, target_part = selected_rel_text.split(" -> ", 1)
            target_id_part, _ = target_part.split(":", 1)
            for i, rel in enumerate(source_mask_obj["relationships"]):
                if rel["label"] == label_part and rel["target_id"] == target_id_part:
                    idx_in_data = i
                    break
        except ValueError:
            messagebox.showerror("Error", "Could not parse the selected relationship.")
            return

        if idx_in_data == -1:
            messagebox.showerror("Error", "Could not find the selected relationship. Please re-select.")
            return

        new_label_val = self.relationship_label_entry.get().strip()
        new_target_display_val = self.target_mask_id_combobox.get()
        if not new_label_val or not new_target_display_val or new_target_display_val == "Select Target Feature":
            messagebox.showwarning("Input Required", "Please provide a new label and target for the update.")
            return

        new_target_id_val = self._target_mask_map.get(new_target_display_val)
        if not new_target_id_val:
            messagebox.showerror("Internal Error", "Cannot map new target display to an ID.")
            return

        if self.update_relationship(self.current_image, self.selected_mask_id, idx_in_data, new_label_val,
                                    new_target_id_val):
            self.display_selected_mask_details()
            self.clear_relationship_input()
            self.status_label.configure(text="Relationship updated successfully.")
            self.update_statistics()

    def on_delete_relationship(self):
        is_rel_selected = any(btn.cget("fg_color") != "transparent" for btn in self.relationship_list_buttons)
        if not is_rel_selected:
            messagebox.showwarning("Selection Required", "Select a relationship from the list to delete.")
            return

        selected_rel_text = ""
        for btn in self.relationship_list_buttons:
            if btn.cget("fg_color") != "transparent":
                selected_rel_text = btn.cget("text")
                break

        source_mask_obj = self.get_mask_by_id(self.current_image, self.selected_mask_id)
        if not source_mask_obj: return

        idx_in_data = -1
        try:
            label_part, target_part = selected_rel_text.split(" -> ", 1)
            target_id_part, _ = target_part.split(":", 1)
            for i, rel in enumerate(source_mask_obj["relationships"]):
                if rel["label"] == label_part and rel["target_id"] == target_id_part:
                    idx_in_data = i
                    break
        except ValueError:
            messagebox.showerror("Error", "Could not parse the selected relationship.")
            return

        if idx_in_data == -1:
            messagebox.showerror("Error", "Could not find the selected relationship. Please re-select.")
            return

        if messagebox.askyesno("Confirm Delete",
                               f"Are you sure you want to delete this relationship?\n\n{selected_rel_text}"):
            if self.delete_relationship(self.current_image, self.selected_mask_id, idx_in_data):
                self.display_selected_mask_details()
                self.clear_relationship_input()
                self.status_label.configure(text="Relationship deleted.")
                self.update_statistics()
            else:
                messagebox.showerror("Error", "Failed to delete the relationship from data.")

    def clear_relationship_input(self):
        self.relationship_label_entry.delete(0, "end")
        self.target_mask_id_combobox.set("Select Target Feature")
        for btn in self.relationship_list_buttons:
            btn.configure(fg_color="transparent")

        if self.selected_target_mask_id is not None:
            self.selected_target_mask_id = None
            self.redraw_image_with_masks()
        self.update_button_states()

    def update_undo_redo_buttons(self):
        undo_state = "normal" if self.undo_stack else "disabled"
        redo_state = "normal" if self.redo_stack else "disabled"
        try:
            self.editmenu.entryconfig("Undo", state=undo_state)
            self.editmenu.entryconfig("Redo", state=redo_state)
        except Exception:
            pass

    # ===================================================================
    # UNCHANGED HELPER AND LOGIC FUNCTIONS
    # ===================================================================
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
            return np.zeros((height, width), dtype=np.uint8)

    def load_cvat_xml(self, xml_file):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            image_data = {}
        except ET.ParseError as e:
            raise
        for image_elem in root.findall("image"):
            image_filename = image_elem.get("name")
            if not image_filename: continue
            image_width = int(image_elem.get("width", 0))
            image_height = int(image_elem.get("height", 0))
            image_id_from_xml = image_elem.get("id")
            if image_width <= 0 or image_height <= 0: continue
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
                    "id": internal_id, "label": elem.get("label", "Unknown"), "attributes": {},
                    "relationships": [], "elem_type": elem.tag, "xml_element": elem
                }
                for attr_elem in elem.findall("attribute"):
                    attr_name = attr_elem.get("name")
                    if attr_name: mask_data["attributes"][attr_name] = attr_elem.text

                for rel_elem in elem.findall("relationship"):
                    target_xml_id = rel_elem.get("with")
                    rel_label_text = rel_elem.text
                    if target_xml_id and rel_label_text:
                        image_data[image_filename]["relationships_from_xml"].append({
                            "source_internal_id_placeholder": internal_id,
                            "target_xml_id": target_xml_id, "label": rel_label_text.strip()
                        })
                bbox_coords = None
                if elem.tag == "mask":
                    mask_data["rle"] = elem.get("rle")
                    try:
                        left, top, width, height = map(float,
                                                       [elem.get("left", 0), elem.get("top", 0), elem.get("width", 0),
                                                        elem.get("height", 0)])
                    except ValueError:
                        left, top, width, height = 0.0, 0.0, 0.0, 0.0
                    if width > 0 and height > 0: bbox_coords = (left, top, width, height)
                elif elem.tag == "box":
                    try:
                        xtl, ytl, xbr, ybr = map(float, [elem.get("xtl", 0), elem.get("ytl", 0), elem.get("xbr", 0),
                                                         elem.get("ybr", 0)])
                    except ValueError:
                        xtl, ytl, xbr, ybr = 0.0, 0.0, 0.0, 0.0
                    width, height = xbr - xtl, ybr - ytl
                    if width > 0 and height > 0: bbox_coords = (xtl, ytl, width, height)
                elif elem.tag == "polygon":
                    points_str = elem.get("points")
                    if points_str:
                        try:
                            points = [float(p) for p in points_str.replace(';', ',').split(",")]
                            mask_data["points"] = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
                            if mask_data["points"]:
                                xs, ys = [p[0] for p in mask_data["points"]], [p[1] for p in mask_data["points"]]
                                min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
                                width, height = max_x - min_x, max_y - min_y
                                if width > 0 and height > 0: bbox_coords = (min_x, min_y, width, height)
                        except (ValueError, IndexError):
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
                    if not source_mask_object: continue
                    target_internal_id = None
                    try:
                        potential_target_mask = next((m for m in data["masks"] if m["id"] == target_xml_id_from_file),
                                                     None)
                        if potential_target_mask: target_internal_id = potential_target_mask["id"]
                    except ValueError:
                        pass
                    if target_internal_id:
                        source_mask_object["relationships"].append(
                            {"label": rel_from_xml["label"], "target_id": target_internal_id})
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
            if image_filename not in image_data_param: continue
            current_image_internal_data = image_data_param[image_filename]
            internal_id_to_saved_xml_id_map = {}
            internal_id_to_new_xml_element_map = {}
            annotation_xml_elements_in_new_tree = [elem for elem in image_elem_in_new_tree.findall("*") if
                                                   elem.tag in ["mask", "box", "polygon", "tag"]]
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

            for internal_mask_item_data in current_image_internal_data["masks"]:
                source_xml_element_new_tree = internal_id_to_new_xml_element_map.get(internal_mask_item_data['id'])
                if not ET.iselement(source_xml_element_new_tree): continue
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
        try:
            import xml.dom.minidom
            xml_str = ET.tostring(root, encoding='unicode')
            dom = xml.dom.minidom.parseString(xml_str)
            pretty_xml_as_string = dom.toprettyxml(indent="  ", encoding="utf-8")
            with open(output_xml_file, "wb") as f:
                f.write(pretty_xml_as_string)
        except ImportError:
            tree.write(output_xml_file, encoding="utf-8", xml_declaration=True)
        except Exception as e:
            messagebox.showerror("XML Write Error", f"Error during XML writing: {e}")
            try:
                tree.write(output_xml_file, encoding="utf-8", xml_declaration=True)
            except Exception as write_e:
                messagebox.showerror("XML Write Error", f"Fallback XML write also failed: {write_e}"); raise write_e

    def load_relationship_labels(self):
        try:
            script_dir_path = os.path.dirname(os.path.abspath(__file__)); file_path_to_load = os.path.join(
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
            self.relationship_labels = set()

    def save_relationship_labels(self):
        try:
            script_dir_path = os.path.dirname(os.path.abspath(__file__)); file_path_to_save = os.path.join(
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

    def undo(self):
        if self.undo_stack:
            action_to_undo = self.undo_stack.pop();
            is_success = self.apply_relationship_action(action_to_undo, reverse=True)
            if is_success:
                self.push_to_redo_stack(action_to_undo);
                self.display_selected_mask_details();
                self.redraw_image_with_masks();
                self.status_label.configure(text=f"Undo: {action_to_undo['type']}")
                self.update_statistics()
            else:
                self.status_label.configure(text="Undo failed")
                self.update_undo_redo_buttons()

    def redo(self):
        if self.redo_stack:
            action_to_redo = self.redo_stack.pop();
            is_success = self.apply_relationship_action(action_to_redo, reverse=False)
            if is_success:
                self.push_to_undo_stack(action_to_redo);
                self.display_selected_mask_details();
                self.redraw_image_with_masks();
                self.status_label.configure(text=f"Redo: {action_to_redo['type']}")
                self.update_statistics()
            else:
                self.status_label.configure(text="Redo failed")
                self.update_undo_redo_buttons()

    def apply_relationship_action(self, action_data, reverse_op=False):
        action_type, img_filename, src_mask_id, rel_idx, original_label, original_target_id = action_data["type"], \
        action_data["image_filename"], action_data["source_mask_id"], action_data["relationship_index"], action_data[
            "label"], action_data["target_id"]
        source_mask_obj = self.get_mask_by_id(img_filename, src_mask_id)
        if not source_mask_obj: return False
        relationships_list = source_mask_obj["relationships"]
        try:
            if action_type == "add":
                item_data = {"label": original_label, "target_id": original_target_id}
                if reverse_op:
                    if item_data in relationships_list: relationships_list.remove(item_data)
                else:
                    relationships_list.append(item_data)
            elif action_type == "update":
                new_label_val, new_target_id_val = action_data["new_label"], action_data["new_target_id"]
                if not (rel_idx is not None and 0 <= rel_idx < len(relationships_list)): return False
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
                        relationships_list.append(item_data_to_restore)
                else:
                    if rel_idx is not None and 0 <= rel_idx < len(relationships_list) and relationships_list[rel_idx][
                        "label"] == original_label and relationships_list[rel_idx]["target_id"] == original_target_id:
                        relationships_list.pop(rel_idx)
                    else:
                        if item_data_to_restore in relationships_list:
                            relationships_list.remove(item_data_to_restore)
                        else:
                            return False
            return True
        except (ValueError, IndexError, KeyError) as e:
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

    def view_knowledge_graph(self):
        if not self.current_image:
            messagebox.showinfo("No Image", "Please select an image first.")
            return

        if not self.loaded_xml_path:
            messagebox.showinfo("No XML", "Please load an XML annotation file first.")
            return

        image_filename = os.path.basename(self.current_image)
        html_file = create_knowledge_graph(self.loaded_xml_path, image_filename)

        if html_file and os.path.exists(html_file):
            webbrowser.open(f'file://{os.path.realpath(html_file)}')
        else:
            messagebox.showerror("Error", f"Could not generate or find the knowledge graph for {image_filename}.")

def main():
    root = ctk.CTk()
    app = SceneGraphAnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
