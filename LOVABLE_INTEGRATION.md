[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "kalshi-weather-ml-service"
version = "1.0.0"
description = "Production-oriented weather probability, calibration, backtesting, and model-registry API for Kalshi-connected applications."
requires-python = ">=3.12,<3.14"
dependencies = [
  "fastapi==0.128.2",
  "uvicorn[standard]==0.48.0",
  "pydantic==2.13.4",
  "pydantic-settings==2.14.1",
  "sqlalchemy==2.0.50",
  "alembic==1.18.4",
  "psycopg[binary]==3.3.3",
  "numpy==2.3.5",
  "pandas==2.2.3",
  "scikit-learn==1.8.0",
  "joblib==1.5.3",
  "httpx==0.28.1",
  "python-multipart==0.0.22",
  "celery==5.6.2",
  "redis==7.4.0",
  "prometheus-client==0.24.1",
  "orjson==3.11.7",
  "boto3==1.42.58",
]

[project.optional-dependencies]
dev = [
  "pytest==9.0.2",
  "pytest-cov==7.0.0",
  "ruff==0.15.5",
  "mypy==1.19.1",
]

[project.scripts]
weather-ml = "app.cli:main"

[tool.setuptools.packages.find]
include = ["app*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --disable-warnings"

[tool.ruff]
target-version = "py312"
line-length = 100
exclude = ["alembic/versions"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]
ignore = ["B008", "E501"]

[tool.mypy]
python_version = "3.12"
plugins = ["pydantic.mypy"]
ignore_missing_imports = true
warn_unused_ignores = true

[tool.coverage.run]
source = ["app"]
omit = ["app/workers/*"]
