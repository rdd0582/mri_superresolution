#!/usr/bin/env python
import subprocess
import curses  # Use windows-curses on Windows
import json

import sys
import os

# Get the project root (directory where launch.py resides)
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the project root to the Python module search path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def get_user_input(stdscr, prompt, default=None, required=False):
    """
    Prompts the user for input with an optional default value.
    If required is True, the user must enter something if no default is provided.
    """
    curses.echo()
    stdscr.addstr(f"{prompt} [{default if default else 'REQUIRED'}]: ")
    stdscr.refresh()
    
    while True:
        try:
            user_input = stdscr.getstr().decode().strip()
        except Exception as e:
            stdscr.addstr("Error reading input. Try again: ")
            stdscr.refresh()
            continue

        if user_input:
            return user_input
        if default is not None:
            return default
        if required:
            stdscr.addstr("Input required. Try again: ")
            stdscr.refresh()

def launch_script(stdscr, script, params, structured=False):
    """
    Clears the screen and runs the specified script with the given params,
    capturing its output in real time and displaying it in the curses UI.
    
    If structured=True, the script is expected to emit JSON-formatted messages.
    Otherwise, plain text output is shown.
    """
    # Insert '-u' flag for unbuffered output.
    command = ['python', '-u', script] + params
    stdscr.clear()
    stdscr.scrollok(True)
    stdscr.addstr("Running command:\n")
    stdscr.addstr("  " + " ".join(command) + "\n\n")
    stdscr.refresh()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,     # Return strings instead of bytes.
        bufsize=1,     # Line-buffered.
        cwd=project_root  # Run the command from the project root.
    )
    
    max_y, max_x = stdscr.getmaxyx()
    
    if structured:
        # Reserve a status window at the top and a log window below.
        status_height = 5
        status_win = stdscr.derwin(status_height, max_x, 0, 0)
        status_win.box()
        log_win = stdscr.derwin(max_y - status_height - 1, max_x, status_height, 0)
        log_win.scrollok(True)
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # If parsing fails, simply append the raw line.
                    log_win.addstr(line)
                    log_win.refresh()
                    continue

                # Handle structured messages.
                msg_type = msg.get("type")
                if msg_type == "batch_update":
                    status_win.clear()
                    status_win.box()
                    status_win.addstr(1, 2, f"Epoch {msg['epoch']} Batch {msg['batch']}/{msg['total_batches']}")
                    status_win.addstr(2, 2, f"Current Loss: {msg['loss']:.4f}")
                    status_win.refresh()
                elif msg_type == "epoch_summary":
                    log_win.addstr(f"Epoch {msg['epoch']} Summary: Avg Loss: {msg['avg_loss']:.4f}, Time: {msg['elapsed']:.2f}s\n")
                    log_win.refresh()
                elif msg_type == "info":
                    log_win.addstr(f"INFO: {msg['message']}\n")
                    log_win.refresh()
                elif msg_type == "params":
                    log_win.addstr("Training Parameters:\n")
                    for key, value in msg.items():
                        if key != "type":
                            log_win.addstr(f"  {key}: {value}\n")
                    log_win.refresh()
                else:
                    log_win.addstr(line)
                    log_win.refresh()
    else:
        # Fallback: plain text output (non-structured logging).
        row = 4
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                try:
                    stdscr.addstr(row, 0, line.rstrip() + "\n")
                except curses.error:
                    pass
                stdscr.refresh()
                row += 1
                # If output exceeds the screen height, scroll.
                if row >= max_y - 2:
                    stdscr.scroll(1)
                    row = max_y - 2

    # Check subprocess return code and notify if nonzero.
    retcode = process.poll()
    if retcode not in (0, None):
        stdscr.addstr(f"\nProcess exited with code {retcode}\n")
        stdscr.refresh()

    # Wait for key press before returning.
    stdscr.addstr(max_y - 1, 0, "Press any key to continue...")
    stdscr.refresh()
    stdscr.getch()

def handle_selection(stdscr, selection):
    """
    Calls the appropriate logic based on which menu item is selected.
    """
    if selection == "Extract Full-Resolution Dataset":
        datasets_dir = get_user_input(stdscr, "Dataset directory", "./datasets")
        output_dir = get_user_input(stdscr, "Output directory", "./training_data")
        # Updated argument name to --datasets_dir
        launch_script(stdscr, "scripts/extract_full_res.py", [
            "--datasets_dir", datasets_dir,
            "--output_dir", output_dir
        ], structured=False)
    elif selection == "Extract Downsampled Dataset":
        datasets_dir = get_user_input(stdscr, "Dataset directory", "./datasets")
        output_dir = get_user_input(stdscr, "Output directory", "./training_data_1.5T")
        # Updated argument name to --datasets_dir
        launch_script(stdscr, "scripts/downsample_extract.py", [
            "--datasets_dir", datasets_dir,
            "--output_dir", output_dir
        ], structured=False)
    elif selection == "Train Model":
        full_res_dir = get_user_input(stdscr, "Full-res dataset directory", "./training_data")
        low_res_dir = get_user_input(stdscr, "Low-res dataset directory", "./training_data_1.5T")
        model_type = get_user_input(stdscr, "Model type (simple/edsr)", "simple", required=True)
        batch_size = get_user_input(stdscr, "Batch size", "16")
        epochs = get_user_input(stdscr, "Epochs", "10")
        learning_rate = get_user_input(stdscr, "Learning rate", "1e-3")
        checkpoint_dir = get_user_input(stdscr, "Checkpoint directory", "./checkpoints")
        
        # Build the list of parameters.
        params = [
            "--full_res_dir", full_res_dir,
            "--low_res_dir", low_res_dir,
            "--batch_size", batch_size,
            "--epochs", epochs,
            "--learning_rate", learning_rate,
            "--checkpoint_dir", checkpoint_dir,
            "--model_type", model_type,
        ]
        if model_type.lower() == "edsr":
            scale = get_user_input(stdscr, "Scale factor (EDSR)", "1")
            params += ["--scale", scale]
        
        # For training, we expect structured logging (JSON messages).
        launch_script(stdscr, "scripts/train.py", params, structured=True)
    elif selection == "Infer Image":
        input_image = get_user_input(stdscr, "Input image path", required=True)
        output_image = get_user_input(stdscr, "Output image path", "output.png")
        model_type = get_user_input(stdscr, "Model type (simple/edsr)", "simple", required=True)
        checkpoint_dir = get_user_input(stdscr, "Checkpoint directory", "./checkpoints")
        launch_script(stdscr, "scripts/infer.py", [
            "--input_image", input_image,
            "--output_image", output_image,
            "--model_type", model_type,
            "--checkpoint_dir", checkpoint_dir
        ], structured=False)
    elif selection == "Exit":
        raise SystemExit

def menu(stdscr):
    """
    The main menu loop that displays options and handles user navigation.
    """
    curses.curs_set(0)  # Hide the cursor
    options = [
        "Extract Full-Resolution Dataset",
        "Extract Downsampled Dataset",
        "Train Model",
        "Infer Image",
        "Exit"
    ]
    selected_idx = 0

    while True:
        stdscr.clear()
        stdscr.addstr("MRI Superresolution Launcher\n\n", curses.A_BOLD)
        for i, option in enumerate(options):
            if i == selected_idx:
                stdscr.addstr(f"> {option}\n", curses.A_REVERSE)
            else:
                stdscr.addstr(f"  {option}\n")
        stdscr.refresh()

        key = stdscr.getch()
        if key in [ord('q'), 27]:  # Allow 'q' or ESC to exit quickly.
            break
        if key == curses.KEY_UP and selected_idx > 0:
            selected_idx -= 1
        elif key == curses.KEY_DOWN and selected_idx < len(options) - 1:
            selected_idx += 1
        elif key in [10, 13]:  # Enter key
            try:
                handle_selection(stdscr, options[selected_idx])
            except SystemExit:
                break

if __name__ == '__main__':
    try:
        import windows_curses as curses  # For Windows
    except ImportError:
        import curses

    curses.wrapper(menu)
