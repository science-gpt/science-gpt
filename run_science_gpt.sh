mkdir ./app/data # dir must exist for dockerfile
docker compose -f docker-compose.yaml -f docker-compose-milvus.yaml up --pull always
