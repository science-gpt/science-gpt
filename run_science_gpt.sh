mkdir ./app/data # dir must exist for dockerfile

DEV_MODE=false
BUILD=false

# Parse arguments
for arg in "$@"
do
    case $arg in
        --dev)
        DEV_MODE=true
        ;;
        --build)
        BUILD=true
        ;;
    esac
done

if [ "$DEV_MODE" = true ] && [ "$BUILD" = true ]; then
    sudo DEV_MODE=true docker compose -f docker-compose-sciencegpt.yaml -f docker-compose-milvus.yaml up --pull always --build
elif [ "$DEV_MODE" = true ]; then
    DEV_MODE=true docker compose -f docker-compose-sciencegpt.yaml -f docker-compose-milvus.yaml up --pull always
elif [ "$BUILD" = true ]; then
    sudo docker compose -f docker-compose-sciencegpt.yaml -f docker-compose-milvus.yaml up --pull always --build
else
    docker compose -f docker-compose-sciencegpt.yaml -f docker-compose-milvus.yaml up --pull always
fi
