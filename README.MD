```markdown
# VEXOO API

Welcome to the VEXOO API, a FastAPI-based backend designed to provide comprehensive search and language model processing capabilities. This project integrates various services, including web crawling and language model generation, to deliver rich, contextual responses to user queries.

## Features

- **Search Engine Integration**: Leverages custom web crawlers to fetch relevant information from the internet based on user queries.
- **Language Model Processing**: Utilizes Mixtral 22B - Azure AI to generate comprehensive responses, summaries, and insights from the collected data.
- **FastAPI Framework**: Built with FastAPI, offering high performance and easy scalability for web applications.
- **Environment Configuration**: Supports `.env` for easy configuration of API keys, endpoints, and other sensitive information.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```sh
git clone https://github.com/your-repository/VEXOO-API.git
cd VEXOO-API
pip install -r requirements.txt
uvicorn app.main:app --reload --proxy-headers --host 0.0.0.0 --port 80
```

```sh
docker build --pull --rm -f "Dockerfile.dev" -t vexoobackend:1.0.2 "."
docker compose -f "docker-compose.yml" up -d --build 

http://localhost:8000/docs#/
```

### Usage

The API provides endpoints for various functionalities, including:

- **GET /**: The root endpoint, providing a brief description and the running time of the API.
- **POST /api/v1/searchEvidence**: Fetches search engine results based on the provided query.
- **POST /api/v1/searchVexoo**: Generates a comprehensive response using the Mixtral22B.
- **POST /api/v1/searchVexooStream**: Generates a comprehensive stream response using the Mixtral22B.

Refer to the API documentation for detailed usage of each endpoint.

## Development

This project is structured as follows:

- **app/**: Contains the FastAPI application, including routers, controllers, and core logic.
- **app/api/v1/**: Version 1 of the API, including endpoints, models, and utility functions.
- **app/api/v1/core/**: Core functionalities, including language model processing.
- **app/api/v1/web_crawler/**: Web crawler implementations for fetching data from the internet.
- **utils/**: Utility functions and classes used across the application.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to discuss proposed changes or report bugs.

## License

This project is licensed under the MIT License - see the LICENSE file for details.