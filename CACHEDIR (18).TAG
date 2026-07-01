services:
  - type: web
    name: kalshi-weather-ml-api
    runtime: docker
    dockerfilePath: ./Dockerfile
    healthCheckPath: /health
    envVars:
      - key: APP_ENV
        value: production
      - key: ML_SERVICE_API_KEYS
        sync: false
      - key: ADMIN_API_KEY
        sync: false
      - key: DATABASE_URL
        fromDatabase:
          name: kalshi-weather-ml-db
          property: connectionString
      - key: REDIS_URL
        fromService:
          type: redis
          name: kalshi-weather-ml-redis
          property: connectionString
      - key: TASK_ALWAYS_EAGER
        value: "false"
      - key: MODEL_STORE_URI
        sync: false
      - key: AWS_REGION
        sync: false
      - key: AWS_ACCESS_KEY_ID
        sync: false
      - key: AWS_SECRET_ACCESS_KEY
        sync: false
      - key: S3_ENDPOINT_URL
        sync: false

  - type: worker
    name: kalshi-weather-ml-worker
    runtime: docker
    dockerfilePath: ./Dockerfile
    dockerCommand: celery -A app.workers.celery_app:celery_app worker --loglevel=INFO --concurrency=2
    envVars:
      - key: APP_ENV
        value: production
      - key: ML_SERVICE_API_KEYS
        sync: false
      - key: ADMIN_API_KEY
        sync: false
      - key: DATABASE_URL
        fromDatabase:
          name: kalshi-weather-ml-db
          property: connectionString
      - key: REDIS_URL
        fromService:
          type: redis
          name: kalshi-weather-ml-redis
          property: connectionString
      - key: TASK_ALWAYS_EAGER
        value: "false"
      - key: MODEL_STORE_URI
        sync: false
      - key: AWS_REGION
        sync: false
      - key: AWS_ACCESS_KEY_ID
        sync: false
      - key: AWS_SECRET_ACCESS_KEY
        sync: false
      - key: S3_ENDPOINT_URL
        sync: false

  - type: redis
    name: kalshi-weather-ml-redis
    ipAllowList: []
    plan: starter

databases:
  - name: kalshi-weather-ml-db
    plan: basic-256mb
