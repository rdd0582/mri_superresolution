#!/usr/bin/env python
import subprocess
import curses  # Use windows-curses on Windows

def get_user_input(stdscr, prompt, default=None, required=False):
    """
    Prompts the user for input with an optional default value.
    If required is True, the user must enter something if no default is provided.
    """
    curses.echo()
    stdscr.addstr(f"{prompt} [{default if default else 'REQUIRED'}]: ")
    stdscr.refresh()
    
    while True:
        user_input = stdscr.getstr().decode().strip()
        if user_input:
            return user_input
        if default is not None:
            return default
        if required:
            stdscr.addstr("Input required. Try again: ")
            stdscr.refresh()

def launch_script(stdscr, script, params):
    """
    Clears the screen, runs the specified script with the given params,
    capturing its output in real time and displaying it in the curses UI.
    Uses unbuffered mode so output appears as it's generated.
    If output exceeds the screen size, scrolling is applied.
    """
    # Insert '-u' flag for unbuffered output.
    command = ['python', '-u', script] + params
    stdscr.clear()
    stdscr.scrollok(True)  # Enable scrolling on the window
    stdscr.addstr("Running command:\n")
    stdscr.addstr("  " + " ".join(command) + "\n\n")
    stdscr.refresh()

    # Launch the subprocess and capture its output in real time.
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,     # Return strings instead of bytes.
        bufsize=1      # Line-buffered.
    )

    # Start displaying output from a given row.
    row = 4
    max_y, _ = stdscr.getmaxyx()

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
            # If the row exceeds the bottom of the window, scroll up.
            if row >= max_y - 2:  # Reserve two lines for prompts.
                stdscr.scroll(1)
                row = max_y - 2

    # Ensure there's room for the prompt.
    if row >= max_y - 1:
        stdscr.scroll(1)
        row = max_y - 2

    stdscr.addstr(row + 1, 0, "Press any key to continue...")
    stdscr.refresh()
    stdscr.getch()

def handle_selection(stdscr, selection):
    """
    Calls the appropriate logic based on which menu item is selected.
    """
    if selection == "Extract Full-Resolution Dataset":
        dataset_dir = get_user_input(stdscr, "Dataset directory", "./dataset")
        output_dir = get_user_input(stdscr, "Output directory", "./training_data")
        launch_script(stdscr, "scripts/extract_full_res.py", [
            "--dataset_dir", dataset_dir,
            "--output_dir", output_dir
        ])
    elif selection == "Extract Downsampled Dataset":
        dataset_dir = get_user_input(stdscr, "Dataset directory", "./dataset")
        output_dir = get_user_input(stdscr, "Output directory", "./training_data_1.5T")
        launch_script(stdscr, "scripts/downsample_extract.py", [
            "--dataset_dir", dataset_dir,
            "--output_dir", output_dir
        ])
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
        # If the model is edsr, prompt for and append the scale parameter.
        if model_type.lower() == "edsr":
            scale = get_user_input(stdscr, "Scale factor (EDSR)", "1")
            params += ["--scale", scale]

        launch_script(stdscr, "scripts/train.py", params)
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
        ])
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
