mkdir ./app/data # dir must exist for dockerfile

if [ "$1" = "--dev" ]; then
    DEV_MODE=true docker compose -f docker-compose-sciencegpt.yaml -f docker-compose-milvus.yaml up --pull always
else
    docker compose -f docker-compose-sciencegpt.yaml -f docker-compose-milvus.yaml up --pull always
fi
