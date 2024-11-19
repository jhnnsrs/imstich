
# ImSwtitch

A Python application for 2D image stitching and stage movement registration using the `arkitekt-next` framework and `ashlarUC2`. This repository includes a Dockerized setup for running the application seamlessly.

## Features

- Image stitching with Ashlar UC2
- Stage movement simulation
- Integration with `arkitekt-next` for server-based workflows
- Docker support for easy deployment

## Requirements

- Docker (for containerized deployment)
- Python 3.10+ (for local development)
- Dependencies listed in `requirements.txt`

## Installation

### Using Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ashlar-stitcher.git
   cd ashlar-stitcher
   ```

2. Build the Docker image:
   ```bash
   docker build -t ashlar-stitcher .
   ```

3. Run the container:
   ```bash
   docker run -p 8000:8000 ashlar-stitcher
   ```

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ashlar-stitcher.git
   cd ashlar-stitcher
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the script:
   ```bash
   python test_ashlar_numpy.py
   ```

## Usage

### Image Stitching

The `stitch2D` function simulates stitching tiles of images with provided position lists and parameters:
- `pixel_size`: Size of a pixel in microns.
- `position_list`: List of x-y positions for the tiles.
- `arrays`: Image arrays in `[tiles, colour, channels, height, width]` format.

### Stage Movement

The `move_stage` function logs axis movements:
- `axis`: The axis to move (default is `"X"`).
- `position`: The position to move to.

### Server Interaction

The script registers these functions with an `arkitekt-next` server for remote interaction. Ensure the server URL is correctly set in the `easy` context manager.

## File Structure

```
ashlar-stitcher/
├── Dockerfile          # Docker setup
├── requirements.txt    # Python dependencies
├── test_ashlar_numpy.py # Main script
├── README.md           # Documentation
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch.
3. Commit changes and push to your fork.
4. Open a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

