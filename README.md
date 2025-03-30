# MRI Super-Resolution Tool

A modern, user-friendly interface for MRI super-resolution tasks, including image extraction, training, and inference.

## Features

- Extract full-resolution MRI images from NIfTI files
- Generate simulated low-resolution images
- Train super-resolution models (UNet, EDSR, or Simple CNN)
- Perform inference on single images
- Modern, color-coded terminal UI
- Progress tracking and status updates
- Comprehensive logging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mri_superresolution.git
cd mri_superresolution
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For Windows users, install windows-curses:
```bash
pip install windows-curses
```

## Usage

1. Start the UI:
```bash
python scripts/ui.py
```

2. Navigate the interface:
- Use Up/Down arrow keys to move between options
- Press Enter to select an option
- Press 'q' to quit
- Type values when prompted for input

3. Workflow:
   a. Extract Full Resolution Images
      - Select input directory containing NIfTI files
      - Select output directory for extracted images
   
   b. Generate Low Resolution Images
      - Select input directory with full-resolution images
      - Select output directory for low-resolution images
      - Set downscale factor
   
   c. Train Model
      - Select model type (UNet, EDSR, or Simple CNN)
      - Configure training parameters
      - Start training
   
   d. Infer on Image
      - Select input image
      - Select model checkpoint
      - Choose output directory

## UI Controls

- **Navigation**: Up/Down arrow keys
- **Selection**: Enter
- **Input Mode**: Enter (when on an input field)
- **Backspace**: Delete last character in input mode
- **Quit**: 'q' key or Ctrl+C

## Logging

All operations are logged to:
- `ui.log`: UI-specific logs
- `training.log`: Training progress and metrics
- `extraction.log`: Image extraction logs

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for full list of dependencies

## License

MIT License - see LICENSE file for details
