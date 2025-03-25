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

# More robust curses import for Windows
try:
    if is_windows:
        # Try multiple import approaches for Windows
        try:
            import curses
        except ImportError:
            try:
                import windows_curses as curses
            except ImportError:
                try:
                    from windows_curses import curses
                except ImportError:
                    raise ImportError("Cannot import curses or windows_curses")
    else:
        import curses
except ImportError as e:
    logger.error(f"Curses library not found: {e}. Please install: pip install windows-curses (Windows) or pip install curses (Unix)")
    print(f"Error: Curses library not found. Please run: pip install windows-curses")
    print(f"If you've already installed it, make sure you're using the same Python environment where it was installed.")
    sys.exit(1)

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
        self.params = {
            # Extract full res params
            "datasets_dir": "./datasets",
            "full_res_output_dir": "./training_data",
            "n_slices_extract": 10,
            "lower_percent": 0.2,
            "upper_percent": 0.8,
            
            # Downsample params
            "downsample_output_dir": "./training_data_1.5T",
            "noise_std": 5,
            "blur_sigma": 0.5,
            
            # Training params
            "full_res_dir": "./training_data",
            "low_res_dir": "./training_data_1.5T",
            "model_type": "unet",
            "base_filters": 32,
            "scale": 1,  # For EDSR model
            "num_features": 64,
            "num_blocks": 8,    # For simple CNN model
            "num_res_blocks": 16,  # For EDSR model
            "batch_size": 8,
            "epochs": 100,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "ssim_weight": 0.7,
            "validation_split": 0.2,
            "patience": 10,
            "num_workers": get_optimal_workers(),  # Set based on system capabilities
            "seed": random.randint(1, 10000),
            "augmentation": False,
            "use_tensorboard": False,
            "use_amp": check_amp_availability(),  # Set based on availability
            "cpu": False,  # Force CPU even if CUDA is available
            "checkpoint_dir": "./checkpoints",
            "log_dir": "./logs",
            
            # Inference params
            "input_image": "",
            "output_image": "output.png",
            "target_image": "",
            "show_comparison": True,
            "show_diff": True,
            "batch_mode": False,
            "save_visualizations": False
        }
        self.status_message = ""
        self.error_message = ""

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
        
        # Draw controls
        controls = "↑/↓: Navigate | Enter: Select | Q: Quit"
        self.stdscr.addstr(height - 1, (width - len(controls)) // 2, controls)
        self.stdscr.attroff(curses.color_pair(1))
        
    def draw_menu(self):
        """Draw the current menu"""
        self.stdscr.clear()
        self.draw_title_bar()
        
        if self.current_menu == "main":
            self.draw_main_menu()
        elif self.current_menu == "extract_full_res":
            self.draw_extract_full_res_menu()
        elif self.current_menu == "downsample":
            self.draw_downsample_menu()
        elif self.current_menu == "train":
            self.draw_train_menu()
        elif self.current_menu == "infer":
            self.draw_infer_menu()
        
        self.draw_status_bar()
        self.stdscr.refresh()
    
    def draw_main_menu(self):
        """Draw the main menu options"""
        self.options = [
            "Extract Full Resolution Images",
            "Generate Low Resolution Images",
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
    
    def draw_extract_full_res_menu(self):
        """Draw the extract full resolution menu"""
        self.options = [
            "datasets_dir",
            "full_res_output_dir",
            "n_slices_extract",
            "lower_percent",
            "upper_percent",
            "Run Extraction",
            "Back to Main Menu"
        ]
        
        height, width = self.stdscr.getmaxyx()
        start_y = 4
        
        # Draw menu title
        self.stdscr.attron(curses.color_pair(2))
        self.stdscr.addstr(2, 2, "Extract Full Resolution Images")
        self.stdscr.attroff(curses.color_pair(2))
        
        # Draw options
        for i, option in enumerate(self.options):
            attr = curses.color_pair(5) | Colors.HIGHLIGHT if i == self.current_option else curses.color_pair(1)
            self.stdscr.attron(attr)
            
            if option not in ["Run Extraction", "Back to Main Menu"]:
                # Format parameter name and value
                param_name = option.ljust(20)
                param_value = str(self.params[option])
                
                self.stdscr.addstr(start_y + i, 4, param_name)
                self.stdscr.attroff(attr)  # Turn off highlight for value
                
                # Use a different color for the parameter value
                self.stdscr.attron(curses.color_pair(6))
                self.stdscr.addstr(start_y + i, 4 + len(param_name) + 2, param_value)
                self.stdscr.attroff(curses.color_pair(6))
            else:
                self.stdscr.addstr(start_y + i, 4, option)
                self.stdscr.attroff(attr)
    
    def draw_downsample_menu(self):
        """Draw the downsample menu"""
        self.options = [
            "datasets_dir",
            "downsample_output_dir",
            "noise_std",
            "blur_sigma",
            "Run Downsampling",
            "Back to Main Menu"
        ]
        
        height, width = self.stdscr.getmaxyx()
        start_y = 4
        
        # Draw menu title
        self.stdscr.attron(curses.color_pair(2))
        self.stdscr.addstr(2, 2, "Generate Low Resolution Images")
        self.stdscr.attroff(curses.color_pair(2))
        
        # Draw options
        for i, option in enumerate(self.options):
            attr = curses.color_pair(5) | Colors.HIGHLIGHT if i == self.current_option else curses.color_pair(1)
            self.stdscr.attron(attr)
            
            if option not in ["Run Downsampling", "Back to Main Menu"]:
                # Format parameter name and value
                param_name = option.ljust(25)
                param_value = str(self.params[option])
                
                self.stdscr.addstr(start_y + i, 4, param_name)
                self.stdscr.attroff(attr)  # Turn off highlight for value
                
                # Use a different color for the parameter value
                self.stdscr.attron(curses.color_pair(6))
                self.stdscr.addstr(start_y + i, 4 + len(param_name) + 2, param_value)
                self.stdscr.attroff(curses.color_pair(6))
            else:
                self.stdscr.addstr(start_y + i, 4, option)
                self.stdscr.attroff(attr)
    
    def draw_train_menu(self):
        """Draw the train menu with all training options"""
        # Define required parameters
        required_params = ["full_res_dir", "low_res_dir", "model_type"]
        
        # Dynamic options based on model_type selection
        common_options = [
            "full_res_dir",
            "low_res_dir",
            "model_type"
        ]
        
        # Model-specific options that should only be shown when the respective model is selected
        unet_options = ["base_filters"] if self.params["model_type"] == "unet" else []
        edsr_options = ["scale", "num_features", "num_res_blocks"] if self.params["model_type"] == "edsr" else []
        simple_options = ["num_features", "num_blocks"] if self.params["model_type"] == "simple" else []
        
        # General training parameters
        training_options = [
            "batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "ssim_weight",
            "validation_split",
            "patience",
            "num_workers",
            "seed"
        ]
        
        # Boolean flags
        flag_options = [
            "augmentation",
            "use_tensorboard",
            "use_amp",
            "cpu"
        ]
        
        # Directory options
        dir_options = [
            "checkpoint_dir",
            "log_dir"
        ]
        
        # Combine all relevant options based on the selected model type
        self.options = common_options + unet_options + edsr_options + simple_options + training_options + flag_options + dir_options + ["Run Training", "Back to Main Menu"]
        
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
        self.stdscr.addstr(3, 2, "* Required parameters")
        self.stdscr.attroff(curses.color_pair(4))
        
        # Show scroll indicators if needed
        if start_index > 0:
            self.stdscr.addstr(3, width // 2, "↑ More options above")
        
        # Draw options
        for i in range(min(visible_options, len(self.options))):
            option_index = start_index + i
            if option_index >= len(self.options):
                break
                
            option = self.options[option_index]
            attr = curses.color_pair(5) | Colors.HIGHLIGHT if option_index == self.current_option else curses.color_pair(1)
            self.stdscr.attron(attr)
            
            if option not in ["Run Training", "Back to Main Menu"]:
                # Format parameter name and value
                param_name = option
                if option in required_params:
                    param_name = f"{option}*"
                param_name = param_name.ljust(21)  # Extra space for *
                param_value = str(self.params[option])
                
                # Show different formatting for boolean options
                if option in flag_options:
                    param_value = "Enabled" if self.params[option] else "Disabled"
                
                self.stdscr.addstr(4 + i, 4, param_name)
                self.stdscr.attroff(attr)  # Turn off highlight for value
                
                # Use a different color for the parameter value
                self.stdscr.attron(curses.color_pair(6))
                self.stdscr.addstr(4 + i, 4 + len(param_name) + 2, param_value)
                self.stdscr.attroff(curses.color_pair(6))
            else:
                self.stdscr.addstr(4 + i, 4, option)
                self.stdscr.attroff(attr)
        
        # Show scroll indicators if needed
        if start_index + visible_options < len(self.options):
            self.stdscr.addstr(height - 4, width // 2, "↓ More options below")
    
    def draw_infer_menu(self):
        """Draw the inference menu"""
        # Define required parameters
        required_params = ["input_image", "model_type", "checkpoint_dir"]
        
        # Common parameters
        common_options = [
            "input_image",
            "output_image",
            "target_image",
            "model_type",
            "checkpoint_dir"
        ]
        
        # Model-specific options that should only be shown when the respective model is selected
        unet_options = ["base_filters"] if self.params["model_type"] == "unet" else []
        edsr_options = ["scale", "num_features", "num_res_blocks"] if self.params["model_type"] == "edsr" else []
        simple_options = ["num_features", "num_blocks"] if self.params["model_type"] == "simple" else []
        
        # Display options
        display_options = [
            "show_comparison",
            "show_diff",
            "batch_mode",
            "save_visualizations"
        ]
        
        # Hardware options
        hardware_options = [
            "use_amp",
            "cpu"
        ]
        
        # Combine all relevant options based on the selected model type
        self.options = common_options + unet_options + edsr_options + simple_options + display_options + hardware_options + ["Run Inference", "Back to Main Menu"]
        
        height, width = self.stdscr.getmaxyx()
        start_y = 4
        
        # Draw menu title
        self.stdscr.attron(curses.color_pair(2))
        self.stdscr.addstr(2, 2, "Infer on Image")
        self.stdscr.attroff(curses.color_pair(2))
        
        # Draw note about required parameters
        self.stdscr.attron(curses.color_pair(4))
        self.stdscr.addstr(3, 2, "* Required parameters")
        self.stdscr.attroff(curses.color_pair(4))
        
        # Draw options
        for i, option in enumerate(self.options):
            attr = curses.color_pair(5) | Colors.HIGHLIGHT if i == self.current_option else curses.color_pair(1)
            self.stdscr.attron(attr)
            
            if option not in ["Run Inference", "Back to Main Menu"]:
                # Format parameter name and value with * for required parameters
                param_name = option
                if option in required_params:
                    param_name = f"{option}*"
                param_name = param_name.ljust(21)  # Extra space for *
                param_value = str(self.params[option])
                
                self.stdscr.addstr(start_y + i, 4, param_name)
                self.stdscr.attroff(attr)  # Turn off highlight for value
                
                # Use a different color for the parameter value
                self.stdscr.attron(curses.color_pair(6))
                self.stdscr.addstr(start_y + i, 4 + len(param_name) + 2, param_value)
                self.stdscr.attroff(curses.color_pair(6))
            else:
                self.stdscr.addstr(start_y + i, 4, option)
                self.stdscr.attroff(attr)
    
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
        elif key == 10:  # Enter key
            return self.handle_option_selection()
        elif key == ord('q') or key == ord('Q'):
            return False  # Exit the app
            
        return True
    
    def handle_input_mode(self, key):
        """Handle input mode for parameter editing"""
        # Define boolean flags for quick reference
        boolean_flags = ['augmentation', 'use_tensorboard', 'use_amp', 'cpu',
                        'show_comparison', 'show_diff', 'batch_mode', 
                        'save_visualizations']
        
        if key == 10:  # Enter key - confirm input
            # For boolean flags, toggle the value or set based on input
            if self.input_field in boolean_flags:
                if self.input_value.lower() in ['toggle']:
                    self.params[self.input_field] = not self.params[self.input_field]
                else:
                    self.params[self.input_field] = (self.input_value.lower() in ['true', 'yes', 'y', '1'])
            else:
                self.params[self.input_field] = self.input_value
            
                # Convert numeric values to appropriate types
                if self.input_field in ['n_slices_extract', 'base_filters', 'scale', 'num_features', 
                                      'num_blocks', 'num_res_blocks', 'batch_size', 'epochs', 
                                      'num_workers', 'seed']:
                    try:
                        self.params[self.input_field] = int(self.input_value)
                    except ValueError:
                        pass  # Keep as string if not convertible
                        
                elif self.input_field in ['lower_percent', 'upper_percent', 'noise_std', 'blur_sigma',
                                       'learning_rate', 'weight_decay', 'ssim_weight', 'validation_split']:
                    try:
                        self.params[self.input_field] = float(self.input_value)
                    except ValueError:
                        pass  # Keep as string if not convertible
            
            # Exit input mode
            self.input_mode = False
        
        elif key == 27:  # Escape key - cancel input
            self.input_mode = False
        
        elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:  # Backspace
            self.input_value = self.input_value[:-1]
        
        elif 32 <= key <= 126:  # Printable characters
            self.input_value += chr(key)
        
        return True

    def handle_option_selection(self):
        """Handle the selection of an option"""
        selected_option = self.options[self.current_option]
        
        # Define boolean flags for quick reference
        boolean_flags = ['augmentation', 'use_tensorboard', 'use_amp', 'cpu',
                        'show_comparison', 'show_diff', 'batch_mode', 
                        'save_visualizations']
        
        # Main menu selections
        if self.current_menu == "main":
            if selected_option == "Extract Full Resolution Images":
                self.current_menu = "extract_full_res"
                self.current_option = 0
            
            elif selected_option == "Generate Low Resolution Images":
                self.current_menu = "downsample"
                self.current_option = 0
            
            elif selected_option == "Train Super-Resolution Model":
                self.current_menu = "train"
                self.current_option = 0
            
            elif selected_option == "Infer on Image":
                self.current_menu = "infer"
                self.current_option = 0
            
            elif selected_option == "Exit":
                return False  # Exit the app
        
        # Extract full res menu selections
        elif self.current_menu == "extract_full_res":
            if selected_option == "Run Extraction":
                self.run_extract_full_res()
            
            elif selected_option == "Back to Main Menu":
                self.current_menu = "main"
                self.current_option = 0
            
            else:
                # Enter editing mode for the parameter
                self.input_mode = True
                self.input_field = selected_option
                self.input_value = str(self.params[selected_option])
                
                if selected_option in boolean_flags:
                    self.input_prompt = f"Toggle {selected_option} (type 'toggle' or yes/no):"
                else:
                    self.input_prompt = f"Enter value for {selected_option}:"
        
        # Downsample menu selections
        elif self.current_menu == "downsample":
            if selected_option == "Run Downsampling":
                self.run_downsample()
            
            elif selected_option == "Back to Main Menu":
                self.current_menu = "main"
                self.current_option = 0
            
            else:
                # Enter editing mode for the parameter
                self.input_mode = True
                self.input_field = selected_option
                self.input_value = str(self.params[selected_option])
                
                if selected_option in boolean_flags:
                    self.input_prompt = f"Toggle {selected_option} (type 'toggle' or yes/no):"
                else:
                    self.input_prompt = f"Enter value for {selected_option}:"
        
        # Train menu selections
        elif self.current_menu == "train":
            if selected_option == "Run Training":
                self.run_training()
            
            elif selected_option == "Back to Main Menu":
                self.current_menu = "main"
                self.current_option = 0
            
            else:
                # Enter editing mode for the parameter
                self.input_mode = True
                self.input_field = selected_option
                self.input_value = str(self.params[selected_option])
                
                if selected_option in boolean_flags:
                    self.input_prompt = f"Toggle {selected_option} (type 'toggle' or yes/no):"
                else:
                    self.input_prompt = f"Enter value for {selected_option}:"
        
        # Infer menu selections
        elif self.current_menu == "infer":
            if selected_option == "Run Inference":
                self.run_inference()
            
            elif selected_option == "Back to Main Menu":
                self.current_menu = "main"
                self.current_option = 0
            
            else:
                # Enter editing mode for the parameter
                self.input_mode = True
                self.input_field = selected_option
                self.input_value = str(self.params[selected_option])
                self.input_prompt = f"Enter value for {selected_option}:"
        
        return True
    
    def run_extract_full_res(self):
        """Run the extract full resolution script"""
        self.status_message = "Extracting full resolution images..."
        self.draw_menu()
        
        try:
            cmd = [
                sys.executable, 
                os.path.join(project_root, "scripts", "extract_full_res.py"),
                "--datasets_dir", self.params["datasets_dir"],
                "--output_dir", self.params["full_res_output_dir"],
                "--n_slices", str(self.params["n_slices_extract"]),
                "--lower_percent", str(self.params["lower_percent"]),
                "--upper_percent", str(self.params["upper_percent"])
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Temporarily exit curses mode
            curses.endwin()
            
            # Run the process with real-time output
            print("\nRunning extract_full_res.py...\n")
            process = subprocess.run(cmd, check=False)
            print("\nPress any key to return to the UI...")
            input()
            
            # Resume curses mode
            self.stdscr = curses.initscr()
            self.init_curses()
            
            if process.returncode == 0:
                self.status_message = "Full resolution extraction completed successfully!"
                self.error_message = ""
            else:
                self.error_message = f"Error during extraction! Check ui.log for details."
                self.status_message = ""
        
        except Exception as e:
            logger.error(f"Error running extract_full_res.py: {e}")
            self.error_message = f"Error: {str(e)}"
            self.status_message = ""
            
            # Resume curses mode if exception occurs
            self.stdscr = curses.initscr()
            self.init_curses()
    
    def run_downsample(self):
        """Run the downsample script"""
        self.status_message = "Generating low resolution images..."
        self.draw_menu()
        
        try:
            cmd = [
                sys.executable, 
                os.path.join(project_root, "scripts", "downsample_extract.py"),
                "--datasets_dir", self.params["datasets_dir"],
                "--output_dir", self.params["downsample_output_dir"],
                "--noise_std", str(self.params["noise_std"]),
                "--blur_sigma", str(self.params["blur_sigma"])
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Temporarily exit curses mode
            curses.endwin()
            
            # Run the process with real-time output
            print("\nRunning downsample_extract.py...\n")
            process = subprocess.run(cmd, check=False)
            print("\nPress any key to return to the UI...")
            input()
            
            # Resume curses mode
            self.stdscr = curses.initscr()
            self.init_curses()
            
            if process.returncode == 0:
                self.status_message = "Low resolution image generation completed successfully!"
                self.error_message = ""
            else:
                self.error_message = f"Error during downsampling! Check ui.log for details."
                self.status_message = ""
        
        except Exception as e:
            logger.error(f"Error running downsample_extract.py: {e}")
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
            if self.params["model_type"] == "unet":
                cmd.extend([
                    "--base_filters", str(self.params["base_filters"])
                ])
            elif self.params["model_type"] == "edsr":
                cmd.extend([
                    "--scale", str(self.params["scale"]),
                    "--num_features", str(self.params["num_features"]),
                    "--num_res_blocks", str(self.params["num_res_blocks"])
                ])
            elif self.params["model_type"] == "simple":
                cmd.extend([
                    "--num_features", str(self.params["num_features"]),
                    "--num_blocks", str(self.params["num_blocks"])
                ])
            
            # Add common training parameters
            cmd.extend([
                "--batch_size", str(self.params["batch_size"]),
                "--epochs", str(self.params["epochs"]),
                "--learning_rate", str(self.params["learning_rate"]),
                "--weight_decay", str(self.params["weight_decay"]),
                "--ssim_weight", str(self.params["ssim_weight"]),
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
            
            # Add model-specific parameters based on the selected model
            if self.params["model_type"] == "unet":
                cmd.extend([
                    "--base_filters", str(self.params["base_filters"])
                ])
            elif self.params["model_type"] == "edsr":
                cmd.extend([
                    "--scale", str(self.params["scale"]),
                    "--num_features", str(self.params["num_features"]),
                    "--num_res_blocks", str(self.params["num_res_blocks"])
                ])
            elif self.params["model_type"] == "simple":
                cmd.extend([
                    "--num_features", str(self.params["num_features"]),
                    "--num_blocks", str(self.params["num_blocks"])
                ])
            
            # Add target if provided
            if self.params["target_image"]:
                cmd.extend(["--target", self.params["target_image"]])
            
            # Add boolean flags
            if self.params["show_comparison"]:
                cmd.append("--show_comparison")
            if self.params["show_diff"]:
                cmd.append("--show_diff")
            if self.params["batch_mode"]:
                cmd.append("--batch_mode")
            if self.params["save_visualizations"]:
                cmd.append("--save_visualizations")
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
                    self.stdscr.addstr(height - 5, 2, prompt)
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