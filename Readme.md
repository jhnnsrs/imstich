
# ImSwtitch

A Python application for 2D image stitching and stage movement registration using the `arkitekt-next` framework and `ashlarUC2`. This repository includes a Dockerized setup for running the application seamlessly.

## Features

- Image stitching with Ashlar UC2
- Creating fake images for testin
- Integration with `arkitekt-next` for server-based workflows
- Docker support for easy deployment

## Requirements

- Docker (for containerized deployment)
- Python 3.12+ (for local development)
- Dependencies listed in `requirements.txt`

## Installation

### Using Devcontainer

This repository comes with a devcontainer setup for easy deployment. To run the application in a container, open the repository in Visual Studio Code and click on the "Reopen in Container" button. Otherwise you will find the Dockerfile in the `.arkitekt_next/flavours/vanilla` directory, which is build
to run the application.

Then in the container terminal, run the following command to start the application:

```bash
arkitekt-next run dev
```

### Using Python

1. Clone the repository:


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
   arkitekt-next run dev
   ```


## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch.
3. Commit changes and push to your fork.
4. Open a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

