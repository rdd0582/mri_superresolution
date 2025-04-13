#!/usr/bin/env python
import os
import sys
import subprocess
import logging
import json
import random
from pathlib import Path

# Add the project root directory to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

"""
MRI Super-Resolution Tool UI

This UI provides an interface to the MRI Super-Resolution pipeline:
1. Extract full resolution images from datasets
2. Generate low resolution images by downsampling
3. Train super-resolution models (UNet, EDSR, or Simple CNN)
4. Run inference on new images

Features:
- Model-specific parameter handling: Only shows and passes parameters relevant to the selected model
- Required parameters are marked with an asterisk (*)
- All operations can be configured and executed from the UI
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Determine if we're on Windows and import the appropriate curses library
is_windows = sys.platform.startswith('win')

import curses

# Import colorama for Windows color support
try:
    from colorama import init, Fore, Back, Style
    init()  # Initialize colorama
    USE_COLORAMA = True
except ImportError:
    USE_COLORAMA = False
    logger.warning("Colorama not found. Colors will be limited. Install with: pip install colorama")

# UI Color Constants
class Colors:
    HEADER = curses.A_BOLD
    NORMAL = curses.A_NORMAL
    HIGHLIGHT = curses.A_REVERSE | curses.A_BOLD
    SELECTED = curses.A_UNDERLINE | curses.A_BOLD
    ERROR = curses.A_BOLD

def check_amp_availability():
    """Check if AMP (Automatic Mixed Precision) can be used"""
    try:
        import torch
        # Check if CUDA is available and if the GPU supports AMP
        if torch.cuda.is_available():
            # Get the current device
            device = torch.cuda.current_device()
            # Check if the GPU supports AMP (compute capability >= 7.0)
            if torch.cuda.get_device_capability(device)[0] >= 7:
                return True
        return False
    except ImportError:
        return False

def get_optimal_workers():
    """Get the optimal number of workers based on system CPU count"""
    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        # Use half of available CPU cores, but at least 1 and at most 8
        return max(1, min(8, cpu_count // 2))
    except Exception:
        return 4  # Fallback to a reasonable default

# UI Class
class MRIUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.init_curses()
        self.current_menu = "main"
        self.current_option = 0
        self.options = []
        self.input_mode = False
        self.input_value = ""
        self.input_prompt = ""
        self.input_field = ""
        
        # Define boolean flags for quick reference
        self.boolean_flags = [
            'augmentation', 'use_tensorboard', 'use_amp', 'cpu',
            'show_comparison', 'show_diff'
        ]
        
        # Define parameters with discrete options
        self.discrete_params = {
            'perceptual_loss_type': ['l1', 'l2', 'mse'],
            'vgg_layer_idx': [16, 19, 22, 25, 29, 32, 35, 38, 42, 45, 49],  # Common VGG16 layer indices for perceptual loss
            'use_kspace_simulation': [True, False]  # Option to toggle between k-space and old simulation method
        }
        
        self.params = {
            # Paired extraction params
            "datasets_dir": "./datasets",
            "hr_output_dir": "./training_data",
            "lr_output_dir": "./training_data_1.5T",
            "n_slices_extract": 10,
            "lower_percent": 0.2,
            "upper_percent": 0.8,
            "noise_std": 5,  # This is appropriate for 0-255 range (scaled internally)
            "blur_sigma": 0.5,
            "target_size": "256 256",  # Added target_size parameter
            "kspace_crop_factor": 0.5,  # Added k-space crop factor parameter
            "use_kspace_simulation": True,  # Default to using k-space simulation
            
            # Training params
            "full_res_dir": "./training_data",
            "low_res_dir": "./training_data_1.5T",
            "model_type": "unet",
            "base_filters": 32,
            "batch_size": 8,
            "epochs": 100,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "ssim_weight": 0.7,
            "perceptual_weight": 0.0,
            "vgg_layer_idx": 35,
            "perceptual_loss_type": 'l1',
            "validation_split": 0.2,
            "patience": 10,
            "num_workers": get_optimal_workers(),  # Set based on system capabilities
            "seed": random.randint(1, 10000),
            "augmentation": False,
            "use_tensorboard": False,
            "use_amp": check_amp_availability(),  # Set based on availability
            "cpu": False,  # Force CPU even if CUDA is available
            "checkpoint_dir": "./checkpoints",
            "checkpoint_file": "", # Selected specific checkpoint file
            "log_dir": "./logs",
            
            # Inference params
            "input_image": "",
            "output_image": "output.png",
            "target_image": "",
            "show_comparison": True,
            "show_diff": True,
        }
        self.status_message = ""
        self.error_message = ""
        self.available_checkpoints = []

    def init_curses(self):
        """Initialize curses settings"""
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, -1)    # Default
        curses.init_pair(2, curses.COLOR_BLUE, -1)     # Headers
        curses.init_pair(3, curses.COLOR_GREEN, -1)    # Success
        curses.init_pair(4, curses.COLOR_RED, -1)      # Error
        curses.init_pair(5, curses.COLOR_YELLOW, -1)   # Highlighted
        curses.init_pair(6, curses.COLOR_CYAN, -1)     # Option values
        
        # Hide cursor and disable echo
        curses.curs_set(0)
        curses.noecho()
        
        # Enable keypad for special keys (arrows)
        self.stdscr.keypad(True)
        
        # Disable cursor blink and timeout for getch
        curses.cbreak()
        self.stdscr.timeout(-1)
        
    def draw_title_bar(self):
        """Draw the title bar"""
        height, width = self.stdscr.getmaxyx()
        title = "MRI Super-Resolution Tool"
        self.stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        self.stdscr.addstr(0, (width - len(title)) // 2, title)
        self.stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
        
        # Draw horizontal line
        self.stdscr.attron(curses.color_pair(1))
        self.stdscr.addstr(1, 0, "=" * (width - 1))
        self.stdscr.attroff(curses.color_pair(1))
    
    def draw_status_bar(self):
        """Draw the status bar at the bottom of the screen"""
        height, width = self.stdscr.getmaxyx()
        
        # Error message has priority
        if self.error_message:
            self.stdscr.attron(curses.color_pair(4))
            message = f" ERROR: {self.error_message} "
            self.stdscr.addstr(height - 2, 0, message.ljust(width - 1))
            self.stdscr.attroff(curses.color_pair(4))
        elif self.status_message:
            self.stdscr.attron(curses.color_pair(3))
            message = f" {self.status_message} "
            self.stdscr.addstr(height - 2, 0, message.ljust(width - 1))
            self.stdscr.attroff(curses.color_pair(3))
        
        # Draw horizontal line and controls
        self.stdscr.attron(curses.color_pair(1))
        self.stdscr.addstr(height - 3, 0, "=" * (width - 1))
        
        # Draw controls - updated to mention selection menus
        controls = "↑/↓: Navigate | Enter: Select/Open Menu | Q: Quit"
        # If width permits, add more context about [Select] items
        if width > 70:
            controls = "↑/↓: Navigate | Enter: Select/Open Menu | [Select] indicates dropdown options | Q: Quit"
            
        self.stdscr.addstr(height - 1, max(0, (width - len(controls)) // 2), controls)
        self.stdscr.attroff(curses.color_pair(1))
        
    def draw_menu(self):
        """Draw the current menu"""
        self.stdscr.clear()
        self.draw_title_bar()
        
        if self.current_menu == "main":
            self.draw_main_menu()
        elif self.current_menu == "extract_paired":
            self.draw_extract_paired_menu()
        elif self.current_menu == "train":
            self.draw_train_menu()
        elif self.current_menu == "infer":
            self.draw_infer_menu()
        
        self.draw_status_bar()
        self.stdscr.refresh()
    
    def draw_main_menu(self):
        """Draw the main menu options"""
        self.options = [
            "Extract Paired Slices",
            "Train Super-Resolution Model",
            "Infer on Image",
            "Exit"
        ]
        
        height, width = self.stdscr.getmaxyx()
        start_y = 4
        
        # Draw menu title
        self.stdscr.attron(curses.color_pair(2))
        self.stdscr.addstr(2, 2, "Main Menu")
        self.stdscr.attroff(curses.color_pair(2))
        
        # Draw options
        for i, option in enumerate(self.options):
            attr = curses.color_pair(5) | Colors.HIGHLIGHT if i == self.current_option else curses.color_pair(1)
            self.stdscr.attron(attr)
            self.stdscr.addstr(start_y + i, 4, option)
            self.stdscr.attroff(attr)
    
    def draw_extract_paired_menu(self):
        """Draw the extract paired slices menu"""
        self.options = [
            "datasets_dir",
            "hr_output_dir",
            "lr_output_dir",
            "n_slices_extract",
            "lower_percent",
            "upper_percent",
            "noise_std",
            "blur_sigma",
            "target_size",
            "kspace_crop_factor",
            "use_kspace_simulation",
            "Run Extraction",
            "Back to Main Menu"
        ]
        
        height, width = self.stdscr.getmaxyx()
        start_y = 4
        
        # Draw menu title
        self.stdscr.attron(curses.color_pair(2))
        self.stdscr.addstr(2, 2, "Extract Paired Slices (HR & LR)")
        self.stdscr.attroff(curses.color_pair(2))
        
        # Add simulation explanation
        self.stdscr.attron(curses.color_pair(6))
        self.stdscr.addstr(3, 2, "noise_std: for 0-255 range, will be scaled automatically. Values 1-10 recommended.")
        self.stdscr.attroff(curses.color_pair(6))
        
        # Add interpolation info
        self.stdscr.attron(curses.color_pair(3))
        self.stdscr.addstr(4, 2, "Uses LANCZOS interpolation for HR and CUBIC for LR with mean-value letterbox padding")
        self.stdscr.attroff(curses.color_pair(3))
        
        # Add k-space simulation info
        self.stdscr.attron(curses.color_pair(3))
        self.stdscr.addstr(5, 2, "K-space simulation applies FFT, center crop, IFFT, and proper Rician noise")
        self.stdscr.attroff(curses.color_pair(3))
        
        # --- Start: Scrolling Logic ---
        # Calculate available space (adjust lines used for title, info, status)
        info_lines = 4 # Title, 3 info lines
        footer_lines = 3 # Status bar lines + 1 for padding
        header_footer_lines = info_lines + footer_lines 
        visible_options = height - header_footer_lines - 1 # Subtract 1 for the first option's y offset

        start_index = 0
        if len(self.options) > visible_options:
            start_index = max(0, min(self.current_option - visible_options // 2, len(self.options) - visible_options))
        # --- End: Scrolling Logic ---
        
        # --- Start: Scroll Indicators ---
        option_display_y = 6 # Start drawing options below info lines
        if start_index > 0:
            self.stdscr.addstr(option_display_y, width // 2, "↑ More options above")
            option_display_y += 1 # Push options down if top scroll indicator is shown
        # --- End: Scroll Indicators ---

        # Draw options (within visible range)
        for i in range(min(visible_options, len(self.options))):
            option_index = start_index + i
            if option_index >= len(self.options):
                break

            option = self.options[option_index]
            attr = curses.color_pair(5) | Colors.HIGHLIGHT if option_index == self.current_option else curses.color_pair(1)
            self.stdscr.attron(attr)
            
            # Calculate display y position based on loop index and starting y
            current_display_y = option_display_y + i

            if option not in ["Run Extraction", "Back to Main Menu"]:
                # Format parameter name and value
                param_name = option.ljust(20)
                param_value = str(self.params.get(option, ""))
                
                # Check if this parameter has a selection menu
                has_selection = option in self.boolean_flags or option in self.discrete_params
                
                # Custom display names for certain parameters
                display_name = param_name
                if option == "use_kspace_simulation":
                    display_name = "Simulation Method".ljust(20)
                elif option == "kspace_crop_factor":
                    display_name = "K-space Center (%)".ljust(20)
                
                self.stdscr.addstr(current_display_y, 4, display_name)
                self.stdscr.attroff(attr)  # Turn off highlight for value
                
                # Use a different color for the parameter value
                self.stdscr.attron(curses.color_pair(6))
                if has_selection:
                    # Add [Select] indicator for parameters with dropdown menus
                    if option in self.boolean_flags:
                        display_value = "Enabled" if self.params[option] else "Disabled"
                        self.stdscr.addstr(current_display_y, 4 + len(param_name) + 2, f"{display_value} [Select]")
                    elif option == "use_kspace_simulation":
                        display_value = "K-space" if self.params[option] else "Blur + Noise"
                        self.stdscr.addstr(current_display_y, 4 + len(param_name) + 2, f"{display_value} [Select]")
                    else:
                        self.stdscr.addstr(current_display_y, 4 + len(param_name) + 2, f"{param_value} [Select]")
                else:
                    # Custom display for certain parameters
                    if option == "kspace_crop_factor":
                        try:
                            # Convert crop factor to percentage
                            percentage = float(param_value) * 100
                            self.stdscr.addstr(current_display_y, 4 + len(param_name) + 2, f"{percentage:.0f}%")
                        except (ValueError, TypeError):
                            # Fallback if conversion fails
                            self.stdscr.addstr(current_display_y, 4 + len(param_name) + 2, param_value)
                    else:
                        self.stdscr.addstr(current_display_y, 4 + len(param_name) + 2, param_value)
                self.stdscr.attroff(curses.color_pair(6))
            else:
                self.stdscr.addstr(current_display_y, 4, option)
                self.stdscr.attroff(attr)
        
        # --- Start: Scroll Indicators ---
        # Show bottom scroll indicator if needed
        if start_index + visible_options < len(self.options):
             # Draw below the last drawn option
             self.stdscr.addstr(option_display_y + min(visible_options, len(self.options)), width // 2, "↓ More options below")
        # --- End: Scroll Indicators ---
    
    def draw_train_menu(self):
        """Draw the train menu with all training options"""
        # Define required parameters (model_type is now selected beforehand)
        required_params = ["full_res_dir", "low_res_dir"] 
        
        # Dynamic options based on model_type selection
        common_options = [
            "full_res_dir",
            "low_res_dir",
            # "model_type" <- REMOVED FROM HERE
        ]
        
        # Model-specific options that should only be shown when the respective model is selected
        # This part is now simplified as model_type is fixed before entering
        unet_options = ["base_filters"]
        
        # General training parameters
        training_options = [
            "batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "ssim_weight",
            "perceptual_weight",
            "vgg_layer_idx",
            "perceptual_loss_type",
            "validation_split",
            "patience",
            "num_workers",
            "seed"
        ]
        
        # Use the class's boolean_flags list for consistency
        # Use a subset of boolean flags that are relevant for training
        flag_options = [flag for flag in self.boolean_flags 
                      if flag in ["augmentation", "use_tensorboard", "use_amp", "cpu"]]
        
        # Directory options
        dir_options = [
            "checkpoint_dir",
            "log_dir"
        ]
        
        # Combine all relevant options based on the selected model type
        self.options = common_options + unet_options + training_options + flag_options + dir_options + ["Run Training", "Back to Main Menu"]
        
        height, width = self.stdscr.getmaxyx()
        
        # Calculate available space
        visible_options = height - 8  # Accounting for title and status bars
        start_index = max(0, min(self.current_option - visible_options // 2, len(self.options) - visible_options))
        
        # Draw menu title
        self.stdscr.attron(curses.color_pair(2))
        self.stdscr.addstr(2, 2, "Train Super-Resolution Model")
        self.stdscr.attroff(curses.color_pair(2))
        
        # Draw note about required parameters
        self.stdscr.attron(curses.color_pair(4))
        self.stdscr.addstr(3, 2, "* Required parameters | Set perceptual_weight > 0 to enable VGG Loss")
        self.stdscr.attroff(curses.color_pair(4))
        
        # Add note about derived L1 weight
        self.stdscr.attron(curses.color_pair(6))
        self.stdscr.addstr(4, 2, "Note: L1 weight = 1.0 - ssim_weight - perceptual_weight")
        self.stdscr.attroff(curses.color_pair(6))
        
        # Show scroll indicators if needed
        if start_index > 0:
            self.stdscr.addstr(4, width // 2, "↑ More options above")
        
        # Draw options
        option_display_y = 5
        for i in range(min(visible_options, len(self.options))):
            option_index = start_index + i
            if option_index >= len(self.options):
                break
                
            option = self.options[option_index]
            attr = curses.color_pair(5) | Colors.HIGHLIGHT if option_index == self.current_option else curses.color_pair(1)
            self.stdscr.attron(attr)
            
            if option not in ["Run Training", "Back to Main Menu"]:
                # Format parameter name and value with * for required parameters
                param_name = option
                if option in required_params:
                    param_name = f"{option}*"
                param_name = param_name.ljust(23)
                param_value = str(self.params[option])
                
                # Check if this parameter has a selection menu
                has_selection = option in self.boolean_flags or option in self.discrete_params
                
                self.stdscr.addstr(option_display_y + i, 4, param_name)
                self.stdscr.attroff(attr)  # Turn off highlight for value
                
                # Use a different color for the parameter value
                self.stdscr.attron(curses.color_pair(6))
                if has_selection:
                    # Add [Select] indicator for parameters with dropdown menus
                    if option in self.boolean_flags:
                        display_value = "Enabled" if self.params[option] else "Disabled"
                        self.stdscr.addstr(option_display_y + i, 4 + len(param_name) + 2, f"{display_value} [Select]")
                    else:
                        self.stdscr.addstr(option_display_y + i, 4 + len(param_name) + 2, f"{param_value} [Select]")
                else:
                    self.stdscr.addstr(option_display_y + i, 4 + len(param_name) + 2, param_value)
                self.stdscr.attroff(curses.color_pair(6))
            else:
                self.stdscr.addstr(option_display_y + i, 4, option)
                self.stdscr.attroff(attr)
        
        # Show scroll indicators if needed
        if start_index + visible_options < len(self.options):
            self.stdscr.addstr(height - 4, width // 2, "↓ More options below")
    
    def draw_infer_menu(self):
        """Draw the inference menu"""
        # Define required parameters (model_type selected beforehand)
        required_params = ["input_image", "checkpoint_dir"]
        
        # Common parameters
        common_options = [
            "input_image",
            "output_image",
            "target_image",
            # "model_type", <- REMOVED FROM HERE
            "checkpoint_dir",
            "Select Checkpoint" 
        ]
        
        # Model-specific options that should only be shown when the respective model is selected
        # Simplified as model_type is fixed before entering
        unet_options = ["base_filters"]
        
        # Display options - use a subset of boolean flags that are relevant for display
        display_options = [flag for flag in self.boolean_flags 
                          if flag in ["show_comparison", "show_diff"]]
        
        # Hardware options - use a subset of boolean flags that are relevant for hardware
        hardware_options = [flag for flag in self.boolean_flags 
                           if flag in ["use_amp", "cpu"]]
        
        # Combine all relevant options based on the selected model type
        self.options = common_options + unet_options + display_options + hardware_options + ["Run Inference", "Back to Main Menu"]
        
        height, width = self.stdscr.getmaxyx()
        
        # --- Start: Scrolling Logic ---
        # Calculate available space (adjust lines used for title, info, status)
        info_lines = 2 # Title, 1 info line
        footer_lines = 3 # Status bar lines + 1 for padding
        header_footer_lines = info_lines + footer_lines
        visible_options = height - header_footer_lines - 1 # Subtract 1 for the first option's y offset
        
        start_index = 0
        if len(self.options) > visible_options:
            start_index = max(0, min(self.current_option - visible_options // 2, len(self.options) - visible_options))
        # --- End: Scrolling Logic ---

        # Draw menu title
        self.stdscr.attron(curses.color_pair(2))
        self.stdscr.addstr(2, 2, "Infer on Image")
        self.stdscr.attroff(curses.color_pair(2))
        
        # Draw note about required parameters
        self.stdscr.attron(curses.color_pair(4))
        self.stdscr.addstr(3, 2, "* Required parameters")
        self.stdscr.attroff(curses.color_pair(4))

        # --- Start: Scroll Indicators ---
        option_display_y = 4 # Start drawing options below info lines
        if start_index > 0:
            self.stdscr.addstr(option_display_y, width // 2, "↑ More options above")
            option_display_y += 1 # Push options down if top scroll indicator is shown
        # --- End: Scroll Indicators ---
        
        # Draw options (within visible range)
        for i in range(min(visible_options, len(self.options))):
            option_index = start_index + i
            if option_index >= len(self.options):
                break

            option = self.options[option_index]
            attr = curses.color_pair(5) | Colors.HIGHLIGHT if option_index == self.current_option else curses.color_pair(1)
            self.stdscr.attron(attr)
            
            # Calculate display y position based on loop index and starting y
            current_display_y = option_display_y + i

            if option not in ["Run Inference", "Back to Main Menu", "Select Checkpoint"]:
                # Format parameter name and value with * for required parameters
                param_name = option
                if option in required_params:
                    param_name = f"{option}*"
                param_name = param_name.ljust(21)  # Extra space for *
                param_value = str(self.params[option])
                
                # Check if this parameter has a selection menu
                has_selection = option in self.boolean_flags or option in self.discrete_params
                
                self.stdscr.addstr(current_display_y, 4, param_name)
                self.stdscr.attroff(attr)  # Turn off highlight for value
                
                # Use a different color for the parameter value
                self.stdscr.attron(curses.color_pair(6))
                if has_selection:
                    # Add [Select] indicator for parameters with dropdown menus
                    if option in self.boolean_flags:
                        display_value = "Enabled" if self.params[option] else "Disabled"
                        self.stdscr.addstr(current_display_y, 4 + len(param_name) + 2, f"{display_value} [Select]")
                    else:
                        self.stdscr.addstr(current_display_y, 4 + len(param_name) + 2, f"{param_value} [Select]")
                else:
                    self.stdscr.addstr(current_display_y, 4 + len(param_name) + 2, param_value)
                self.stdscr.attroff(curses.color_pair(6))
            else:
                # If it's the Select Checkpoint option, show the currently selected checkpoint
                if option == "Select Checkpoint":
                    param_name = option.ljust(21)
                    self.stdscr.addstr(current_display_y, 4, param_name)
                    self.stdscr.attroff(attr)
                    
                    # Display the selected checkpoint or a message to select one
                    self.stdscr.attron(curses.color_pair(6))
                    if self.params["checkpoint_file"]:
                        self.stdscr.addstr(current_display_y, 4 + len(param_name) + 2, self.params["checkpoint_file"])
                    else:
                        self.stdscr.addstr(current_display_y, 4 + len(param_name) + 2, "None selected [Select]")
                    self.stdscr.attroff(curses.color_pair(6))
                else:
                    self.stdscr.addstr(current_display_y, 4, option)
                    self.stdscr.attroff(attr)
        
        # --- Start: Scroll Indicators ---
        # Show bottom scroll indicator if needed
        if start_index + visible_options < len(self.options):
            # Draw below the last drawn option
            self.stdscr.addstr(option_display_y + min(visible_options, len(self.options)), width // 2, "↓ More options below")
        # --- End: Scroll Indicators ---
    
    def handle_input(self):
        """Handle user input"""
        key = self.stdscr.getch()
        
        if self.input_mode:
            return self.handle_input_mode(key)
        
        # Regular navigation mode
        if key == curses.KEY_UP:
            self.current_option = max(0, self.current_option - 1)
        elif key == curses.KEY_DOWN:
            self.current_option = min(len(self.options) - 1, self.current_option + 1)
        elif key == curses.KEY_RESIZE:
            # --- Explicit Resize Handling ---
            try:
                # When a resize event occurs, simply clear the screen.
                # The main loop's call to draw_menu() will redraw everything
                # using the updated dimensions fetched by getmaxyx().
                self.stdscr.clear() 
            except curses.error as e:
                # Log potential errors during resize handling but try to continue
                logger.error(f"Error clearing screen on resize: {e}")
            # --- End Explicit Resize Handling ---
        elif key == 10:  # Enter key
            return self.handle_option_selection()
        elif key == ord('q') or key == ord('Q'):
            return False  # Exit the app
            
        return True
    
    def handle_input_mode(self, key):
        """Handle input mode for parameter editing"""
        # Store potential error message
        validation_error = None
        
        if key == 10:  # Enter key - confirm input
            # Boolean flags and discrete options are now handled through the select_from_options method
            # This input mode is now only for free-text/numeric inputs
            original_value = self.params[self.input_field]
            try:
                # Attempt type conversion and validation
                if self.input_field in ['n_slices_extract', 'base_filters', 
                                      'batch_size', 'epochs', 
                                      'patience', 'num_workers', 'seed', 'vgg_layer_idx']: 
                    new_value = int(self.input_value)
                elif self.input_field in ['lower_percent', 'upper_percent', 'noise_std', 'blur_sigma',
                                        'learning_rate', 'weight_decay', 'ssim_weight', 
                                        'perceptual_weight', 'validation_split', 'kspace_crop_factor']:
                    new_value = float(self.input_value)
                    # Validation for weights
                    if self.input_field in ['ssim_weight', 'perceptual_weight']:
                        if not (0 <= new_value <= 1):
                            validation_error = f"{self.input_field} must be between 0.0 and 1.0"
                        else:
                            # Check sum constraint
                            other_weight_field = 'perceptual_weight' if self.input_field == 'ssim_weight' else 'ssim_weight'
                            other_weight_val = float(self.params[other_weight_field]) # Ensure float comparison
                            if new_value + other_weight_val > 1.0:
                                validation_error = f"Sum of ssim_weight ({other_weight_val if self.input_field == 'perceptual_weight' else new_value:.2f}) and perceptual_weight ({new_value if self.input_field == 'perceptual_weight' else other_weight_val:.2f}) cannot exceed 1.0"
                    # Validation for kspace_crop_factor
                    elif self.input_field == 'kspace_crop_factor':
                        if not (0 < new_value <= 1):
                            validation_error = f"K-space crop factor must be between 0 and 1 (0% to 100% of center k-space)"
                else: # String parameters like paths
                    new_value = self.input_value
                    
                # If validation passed, update the parameter
                if validation_error is None:
                    self.params[self.input_field] = new_value
                    self.error_message = "" # Clear previous errors on successful update
                    self.status_message = f"Updated {self.input_field} to {new_value}"
                else:
                    self.error_message = validation_error # Show validation error
                    self.params[self.input_field] = original_value # Revert to original value

            except ValueError:
                # Handle conversion error (simplified expected types)
                expected_type = "integer" if self.input_field in ['n_slices_extract', 'base_filters', 'batch_size', 'epochs', 'patience', 'num_workers', 'seed', 'vgg_layer_idx'] else "float"
                self.error_message = f"Invalid input for {self.input_field}. Expected {expected_type}."
                self.params[self.input_field] = original_value # Revert
                validation_error = self.error_message # Ensure error persists
                
            # Exit input mode only if there was no validation error
            if validation_error is None:
                self.input_mode = False
            else:
                # Keep input mode active to allow correction
                self.input_value = str(self.params[self.input_field]) # Reset input display to (potentially reverted) value
                pass 
        
        elif key == 27:  # Escape key - cancel input
            self.input_mode = False
            self.error_message = "" # Clear error message on cancel
        
        elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:  # Backspace
            self.input_value = self.input_value[:-1]
        
        elif 32 <= key <= 126:  # Printable characters
            self.input_value += chr(key)
        
        return True

    def handle_option_selection(self):
        """Handle the selection of an option"""
        selected_option = self.options[self.current_option]
        
        # Main menu selections
        if self.current_menu == "main":
            if selected_option == "Extract Paired Slices":
                self.current_menu = "extract_paired"
                self.current_option = 0
            
            elif selected_option == "Train Super-Resolution Model":
                # Instead of directly switching, call the model selection screen
                self.select_model_type("train") 
                # select_model_type will set the current_menu if successful
            
            elif selected_option == "Infer on Image":
                 # Instead of directly switching, call the model selection screen
                self.select_model_type("infer")
                 # select_model_type will set the current_menu if successful
            
            elif selected_option == "Exit":
                return False  # Exit the app
        
        # Extract paired slices menu selections
        elif self.current_menu == "extract_paired":
            if selected_option == "Run Extraction":
                self.run_extract_paired_slices()
            
            elif selected_option == "Back to Main Menu":
                self.current_menu = "main"
                self.current_option = 0
            
            else:
                # Check if this is a boolean flag or discrete parameter
                if selected_option in self.boolean_flags:
                    self.select_from_options(selected_option, ["Enabled", "Disabled"])
                elif selected_option in self.discrete_params:
                    self.select_from_options(selected_option, self.discrete_params[selected_option])
                else:
                    # Enter editing mode for other parameters
                    self.input_mode = True
                    self.input_field = selected_option
                    self.input_value = str(self.params[selected_option])
                    self.input_prompt = f"Enter value for {selected_option}:"
        
        # Train menu selections
        elif self.current_menu == "train":
            if selected_option == "Run Training":
                self.run_training()
            
            elif selected_option == "Back to Main Menu":
                self.current_menu = "main"
                self.current_option = 0
            
            else:
                # Check if this is a boolean flag or discrete parameter
                if selected_option in self.boolean_flags:
                    self.select_from_options(selected_option, ["Enabled", "Disabled"])
                elif selected_option in self.discrete_params:
                    self.select_from_options(selected_option, self.discrete_params[selected_option])
                else:
                    # Enter editing mode for other parameters
                    self.input_mode = True
                    self.input_field = selected_option
                    self.input_value = str(self.params[selected_option])
                    self.input_prompt = f"Enter value for {selected_option}:"
        
        # Infer menu selections
        elif self.current_menu == "infer":
            if selected_option == "Run Inference":
                self.run_inference()
            
            elif selected_option == "Back to Main Menu":
                self.current_menu = "main"
                self.current_option = 0
                
            elif selected_option == "Select Checkpoint":
                self.select_checkpoint()
            
            else:
                # Check if this is a boolean flag or discrete parameter
                if selected_option in self.boolean_flags:
                    self.select_from_options(selected_option, ["Enabled", "Disabled"])
                elif selected_option in self.discrete_params:
                    self.select_from_options(selected_option, self.discrete_params[selected_option])
                else:
                    # Enter editing mode for other parameters
                    self.input_mode = True
                    self.input_field = selected_option
                    self.input_value = str(self.params[selected_option])
                    self.input_prompt = f"Enter value for {selected_option}:"
        
        return True
    
    def run_extract_paired_slices(self):
        """Run the extract paired slices script"""
        self.status_message = "Extracting paired slices..."
        self.draw_menu()
        
        try:
            cmd = [
                sys.executable, 
                os.path.join(project_root, "scripts", "extract_paired_slices.py"),
                "--datasets_dir", self.params["datasets_dir"],
                "--hr_output_dir", self.params["hr_output_dir"],
                "--lr_output_dir", self.params["lr_output_dir"],
                "--n_slices", str(self.params["n_slices_extract"]),
                "--lower_percent", str(self.params["lower_percent"]),
                "--upper_percent", str(self.params["upper_percent"]),
                "--noise_std", str(self.params["noise_std"]),
                "--blur_sigma", str(self.params["blur_sigma"]),
                "--kspace_crop_factor", str(self.params["kspace_crop_factor"]),
            ]
            
            # Add use_kspace_simulation flag if set to True
            if self.params["use_kspace_simulation"]:
                cmd.append("--use_kspace_simulation")
            
            # Split target_size into separate arguments
            target_size_values = self.params["target_size"].split()
            cmd.extend(["--target_size"] + target_size_values)
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Temporarily exit curses mode
            curses.endwin()
            
            # Run the process with real-time output
            print("\n=== MRI Paired Slice Extraction ===")
            print(f"Datasets Directory: {self.params['datasets_dir']}")
            print(f"High-Resolution Output: {self.params['hr_output_dir']} (Using LANCZOS interpolation for resizing)")
            if self.params["lr_output_dir"]:
                print(f"Low-Resolution Output: {self.params['lr_output_dir']} (Using CUBIC interpolation for resizing)")
                print(f"Simulation Settings:")
            print("\nRunning extract_paired_slices.py...\n")
            process = subprocess.run(cmd, check=False)
            print("\nPress any key to return to the UI...")
            input()
            
            # Resume curses mode
            self.stdscr = curses.initscr()
            self.init_curses()
            
            if process.returncode == 0:
                self.status_message = "Paired slices extraction completed successfully!"
                self.error_message = ""
            else:
                self.error_message = f"Error during extraction! Check ui.log for details."
                self.status_message = ""
        
        except Exception as e:
            logger.error(f"Error running extract_paired_slices.py: {e}")
            self.error_message = f"Error: {str(e)}"
            self.status_message = ""
            
            # Resume curses mode if exception occurs
            self.stdscr = curses.initscr()
            self.init_curses()
    
    def run_training(self):
        """Run the training script"""
        self.status_message = "Training model..."
        self.draw_menu()
        
        try:
            # Start with common parameters
            cmd = [
                sys.executable, 
                os.path.join(project_root, "scripts", "train.py"),
                "--full_res_dir", self.params["full_res_dir"],
                "--low_res_dir", self.params["low_res_dir"],
                "--model_type", self.params["model_type"]
            ]
            
            # Add model-specific parameters based on the selected model
            cmd.extend([
                "--base_filters", str(self.params["base_filters"])
            ])
            
            # Add common training parameters
            cmd.extend([
                "--batch_size", str(self.params["batch_size"]),
                "--epochs", str(self.params["epochs"]),
                "--learning_rate", str(self.params["learning_rate"]),
                "--weight_decay", str(self.params["weight_decay"]),
                "--ssim_weight", str(self.params["ssim_weight"]),
                "--perceptual_weight", str(self.params["perceptual_weight"]),
                "--vgg_layer_idx", str(self.params["vgg_layer_idx"]),
                "--perceptual_loss_type", self.params["perceptual_loss_type"],
                "--validation_split", str(self.params["validation_split"]),
                "--patience", str(self.params["patience"]),
                "--num_workers", str(self.params["num_workers"]),
                "--seed", str(self.params["seed"]),
                "--checkpoint_dir", self.params["checkpoint_dir"],
                "--log_dir", self.params["log_dir"]
            ])
            
            # Add boolean flags
            if self.params["augmentation"]:
                cmd.append("--augmentation")
            if self.params["use_tensorboard"]:
                cmd.append("--use_tensorboard")
            if self.params["use_amp"]:
                cmd.append("--use_amp")
            if self.params["cpu"]:
                cmd.append("--cpu")
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Temporarily exit curses mode
            curses.endwin()
            
            # Run the process with real-time output
            print("\nRunning train.py...\n")
            process = subprocess.run(cmd, check=False)
            print("\nPress any key to return to the UI...")
            input()
            
            # Resume curses mode
            self.stdscr = curses.initscr()
            self.init_curses()
            
            if process.returncode == 0:
                self.status_message = "Training completed successfully!"
                self.error_message = ""
            else:
                self.error_message = f"Error during training! Check ui.log and training.log for details."
                self.status_message = ""
        
        except Exception as e:
            logger.error(f"Error running train.py: {e}")
            self.error_message = f"Error: {str(e)}"
            self.status_message = ""
            
            # Resume curses mode if exception occurs
            self.stdscr = curses.initscr()
            self.init_curses()
    
    def run_inference(self):
        """Run the inference script"""
        self.status_message = "Running inference..."
        self.draw_menu()
        
        try:
            cmd = [
                sys.executable, 
                os.path.join(project_root, "scripts", "infer.py"),
                "--input", self.params["input_image"],
                "--output", self.params["output_image"],
                "--model_type", self.params["model_type"],
                "--checkpoint_dir", self.params["checkpoint_dir"]
            ]
            
            # Add specific checkpoint file if selected
            if self.params["checkpoint_file"]:
                checkpoint_path = os.path.join(self.params["checkpoint_dir"], self.params["checkpoint_file"])
                cmd.extend(["--checkpoint_path", checkpoint_path])
            
            # Add model-specific parameters based on the selected model
            cmd.extend([
                "--base_filters", str(self.params["base_filters"])
            ])
            
            # Add target if provided
            if self.params["target_image"]:
                cmd.extend(["--target", self.params["target_image"]])
            
            # Add boolean flags
            if self.params["show_comparison"]:
                cmd.append("--show_comparison")
            if self.params["show_diff"]:
                cmd.append("--show_diff")
            if self.params["use_amp"]:
                cmd.append("--use_amp")
            if self.params["cpu"]:
                cmd.append("--cpu")
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Temporarily exit curses mode
            curses.endwin()
            
            # Run the process with real-time output
            print("\nRunning infer.py...\n")
            process = subprocess.run(cmd, check=False)
            print("\nPress any key to return to the UI...")
            input()
            
            # Resume curses mode
            self.stdscr = curses.initscr()
            self.init_curses()
            
            if process.returncode == 0:
                self.status_message = "Inference completed successfully!"
                self.error_message = ""
            else:
                self.error_message = f"Error during inference! Check ui.log and inference.log for details."
                self.status_message = ""
        
        except Exception as e:
            logger.error(f"Error running infer.py: {e}")
            self.error_message = f"Error: {str(e)}"
            self.status_message = ""
            
            # Resume curses mode if exception occurs
            self.stdscr = curses.initscr()
            self.init_curses()
    
    def get_available_checkpoints(self):
        """Get available checkpoints for the selected model type"""
        checkpoint_dir = self.params["checkpoint_dir"]
        model_type = self.params["model_type"]
        
        self.available_checkpoints = []
        
        try:
            if os.path.exists(checkpoint_dir):
                # Look for checkpoints matching the model type (unet only now)
                for file in sorted(os.listdir(checkpoint_dir)):
                    # Simplified check: Just look for .pth files as only UNet is supported
                    # If you want to be stricter, check for 'unet' in filename:
                    # if file.endswith('.pth') and model_type in file:
                    if file.endswith('.pth'): 
                        self.available_checkpoints.append(file)
                
                # Removed fallback logic as we only expect UNet checkpoints now
                # if not self.available_checkpoints:
                #     self.available_checkpoints = [file for file in sorted(os.listdir(checkpoint_dir)) 
                #                                 if file.endswith('.pth')]
            
            return self.available_checkpoints
        except Exception as e:
            logger.error(f"Error getting checkpoints: {e}")
            self.error_message = f"Error getting checkpoints: {str(e)}"
            return []

    def select_checkpoint(self):
        """Display a menu to select a checkpoint file"""
        # Get available checkpoints
        checkpoints = self.get_available_checkpoints()
        
        if not checkpoints:
            self.error_message = f"No checkpoints found in {self.params['checkpoint_dir']}"
            return
        
        # Create a temporary screen for checkpoint selection
        checkpoint_stdscr = curses.newwin(0, 0)
        checkpoint_stdscr.keypad(True)
        
        current_selection = 0
        
        while True:
            checkpoint_stdscr.clear()
            height, width = checkpoint_stdscr.getmaxyx()
            
            # Title
            checkpoint_stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
            title = f"Select Checkpoint for {self.params['model_type']} Model"
            checkpoint_stdscr.addstr(1, (width - len(title)) // 2, title)
            checkpoint_stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            
            # Instructions
            checkpoint_stdscr.attron(curses.color_pair(1))
            checkpoint_stdscr.addstr(3, 2, "↑/↓: Navigate | Enter: Select | Esc: Cancel")
            checkpoint_stdscr.attroff(curses.color_pair(1))
            
            # Draw horizontal line
            checkpoint_stdscr.attron(curses.color_pair(1))
            checkpoint_stdscr.addstr(4, 0, "=" * (width - 1))
            checkpoint_stdscr.attroff(curses.color_pair(1))
            
            # Display up to 15 checkpoints at a time with scrolling if needed
            max_display = min(15, height - 10)
            start_idx = max(0, current_selection - max_display // 2)
            end_idx = min(len(checkpoints), start_idx + max_display)
            
            # Adjust start_idx if we have fewer items at the end
            if end_idx - start_idx < max_display and start_idx > 0:
                start_idx = max(0, end_idx - max_display)
            
            # Show scroll indicator
            if start_idx > 0:
                checkpoint_stdscr.addstr(5, width // 2, "↑ More checkpoints above")
            
            # Display checkpoints
            for i in range(start_idx, end_idx):
                attr = curses.color_pair(5) | Colors.HIGHLIGHT if i == current_selection else curses.color_pair(1)
                checkpoint_stdscr.attron(attr)
                checkpoint_stdscr.addstr(6 + i - start_idx, 4, checkpoints[i])
                checkpoint_stdscr.attroff(attr)
            
            # Show scroll indicator
            if end_idx < len(checkpoints):
                checkpoint_stdscr.addstr(6 + end_idx - start_idx, width // 2, "↓ More checkpoints below")
            
            checkpoint_stdscr.refresh()
            
            # Handle input
            key = checkpoint_stdscr.getch()
            
            if key == curses.KEY_UP:
                current_selection = max(0, current_selection - 1)
            elif key == curses.KEY_DOWN:
                current_selection = min(len(checkpoints) - 1, current_selection + 1)
            elif key == 10:  # Enter key
                # Select the checkpoint
                self.params["checkpoint_file"] = checkpoints[current_selection]
                return
            elif key == 27:  # Escape key
                # Cancel
                return
    
    def select_model_type(self, next_menu):
        """Display a menu to select the model type before proceeding."""
        available_models = ['unet'] # Currently only UNet
        
        if not available_models:
            self.error_message = "No models available for selection."
            return # Stay on the current menu

        # Ensure curses is initialized for the new window/screen
        # Using the existing stdscr temporarily might be simpler than newwin if it works
        # Or create a dedicated window like in select_checkpoint
        
        model_select_stdscr = self.stdscr # Use the main screen for simplicity
        model_select_stdscr.keypad(True) # Ensure keypad is enabled
        
        current_selection = 0
        original_menu = self.current_menu
        original_option = self.current_option

        while True:
            model_select_stdscr.clear() # Clear screen for model selection
            self.draw_title_bar() # Keep title consistent
            height, width = model_select_stdscr.getmaxyx()
            
            # Title for this selection screen
            model_select_stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
            title = f"Select Model Architecture"
            model_select_stdscr.addstr(3, (width - len(title)) // 2, title)
            model_select_stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            
            # Instructions
            model_select_stdscr.attron(curses.color_pair(1))
            model_select_stdscr.addstr(5, 2, "↑/↓: Navigate | Enter: Select | Esc: Cancel")
            model_select_stdscr.attroff(curses.color_pair(1))

            start_y = 7
            # Display models
            for i, model_name in enumerate(available_models):
                attr = curses.color_pair(5) | Colors.HIGHLIGHT if i == current_selection else curses.color_pair(1)
                model_select_stdscr.attron(attr)
                model_select_stdscr.addstr(start_y + i, 4, model_name.upper()) # Display in uppercase for clarity
                model_select_stdscr.attroff(attr)
            
            # Also draw status bar at the bottom
            self.draw_status_bar() 
            model_select_stdscr.refresh()
            
            # Handle input
            key = model_select_stdscr.getch()
            
            if key == curses.KEY_UP:
                current_selection = max(0, current_selection - 1)
            elif key == curses.KEY_DOWN:
                current_selection = min(len(available_models) - 1, current_selection + 1)
            elif key == 10:  # Enter key
                # Select the model and proceed to the next menu
                self.params["model_type"] = available_models[current_selection]
                self.current_menu = next_menu # Set the next menu (train or infer)
                self.current_option = 0 # Reset option index for the new menu
                self.error_message = "" # Clear any previous errors
                self.status_message = f"Selected model: {self.params['model_type'].upper()}"
                # Break the loop, the main run loop will redraw the new menu
                break 
            elif key == 27:  # Escape key
                # Cancel and return to the previous menu state
                self.current_menu = original_menu
                self.current_option = original_option
                self.error_message = "Model selection cancelled."
                # Break the loop, the main run loop will redraw the original menu
                break 

        # No explicit return needed, loop breaks and run() continues
    
    def select_from_options(self, param_name, options):
        """Display a menu to select from a list of options for a parameter"""
        # Create a temporary screen for option selection
        option_stdscr = curses.newwin(0, 0)
        option_stdscr.keypad(True)
        
        # Format the current value to select the appropriate option in the list
        current_value = self.params[param_name]
        
        # Set current_selection based on the current value
        if param_name in self.boolean_flags:
            current_selection = 0 if current_value else 1  # 0 for Enabled, 1 for Disabled
        elif param_name == 'use_kspace_simulation':
            # For use_kspace_simulation, we need to handle the boolean value
            current_selection = 0 if current_value else 1  # 0 for True, 1 for False
        else:
            # Try to find the current value in the options list
            try:
                current_selection = options.index(current_value)
            except (ValueError, TypeError):
                # If not found or type mismatch, default to first option
                current_selection = 0
        
        while True:
            option_stdscr.clear()
            height, width = option_stdscr.getmaxyx()
            
            # Title
            option_stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
            title = f"Select Value for {param_name}"
            option_stdscr.addstr(1, (width - len(title)) // 2, title)
            option_stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            
            # Instructions
            option_stdscr.attron(curses.color_pair(1))
            option_stdscr.addstr(3, 2, "↑/↓: Navigate | Enter: Select | Esc: Cancel")
            option_stdscr.attroff(curses.color_pair(1))
            
            # Draw horizontal line
            option_stdscr.attron(curses.color_pair(1))
            option_stdscr.addstr(4, 0, "=" * (width - 1))
            option_stdscr.attroff(curses.color_pair(1))
            
            # Display options with scrolling if needed
            max_display = min(15, height - 10)
            start_idx = max(0, current_selection - max_display // 2)
            end_idx = min(len(options), start_idx + max_display)
            
            # Adjust start_idx if we have fewer items at the end
            if end_idx - start_idx < max_display and start_idx > 0:
                start_idx = max(0, end_idx - max_display)
            
            # Show scroll indicator
            if start_idx > 0:
                option_stdscr.addstr(5, width // 2, "↑ More options above")
            
            # Display options
            for i in range(start_idx, end_idx):
                attr = curses.color_pair(5) | Colors.HIGHLIGHT if i == current_selection else curses.color_pair(1)
                option_stdscr.attron(attr)
                
                # Format display text based on parameter type
                display_text = str(options[i])
                if param_name == 'use_kspace_simulation':
                    # For use_kspace_simulation, show a more descriptive text
                    display_text = 'K-space Simulation' if options[i] else 'Blur + Noise (Legacy)'
                
                option_stdscr.addstr(6 + i - start_idx, 4, display_text)
                option_stdscr.attroff(attr)
            
            # Show scroll indicator
            if end_idx < len(options):
                option_stdscr.addstr(6 + end_idx - start_idx, width // 2, "↓ More options below")
            
            option_stdscr.refresh()
            
            # Handle input
            key = option_stdscr.getch()
            
            if key == curses.KEY_UP:
                current_selection = max(0, current_selection - 1)
            elif key == curses.KEY_DOWN:
                current_selection = min(len(options) - 1, current_selection + 1)
            elif key == 10:  # Enter key
                # Update the parameter value based on selection
                if param_name in self.boolean_flags:
                    # For boolean options, map "Enabled"/"Disabled" to True/False
                    self.params[param_name] = (options[current_selection] == "Enabled")
                elif param_name == 'use_kspace_simulation':
                    # For use_kspace_simulation, set the boolean value directly
                    self.params[param_name] = options[current_selection]
                    display_value = 'K-space Simulation' if options[current_selection] else 'Blur + Noise (Legacy)'
                    self.status_message = f"Updated {param_name} to {display_value}"
                    return
                else:
                    # For non-boolean options, use the selected value
                    self.params[param_name] = options[current_selection]
                self.status_message = f"Updated {param_name} to {options[current_selection]}"
                return
            elif key == 27:  # Escape key
                # Cancel
                return

    def run(self):
        """Main UI loop"""
        try:
            while True:
                self.draw_menu()
                
                # If we're in input mode, draw the input prompt
                if self.input_mode:
                    height, width = self.stdscr.getmaxyx()
                    prompt = f"{self.input_prompt} {self.input_value}"
                    self.stdscr.attron(curses.color_pair(5))
                    self.stdscr.addstr(height - 4, 2, prompt)
                    self.stdscr.attroff(curses.color_pair(5))
                
                if not self.handle_input():
                    break
        
        except Exception as e:
            # Clean exit in case of exception
            curses.endwin()
            logger.error(f"UI Error: {e}")
            print(f"Error: {e}")
            return 1
        
        return 0

def main():
    """Main entry point"""
    try:
        # Initialize colorama for Windows color support
        if USE_COLORAMA:
            init()
        
        # Run the curses application
        return curses.wrapper(lambda stdscr: MRIUI(stdscr).run())
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 