mkdir ./app/data # dir must exist for dockerfile

DEV_MODE=false
BUILD=false
UPDATE_DEPS=false

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
        --update-deps)
        UPDATE_DEPS=true
        ;;
    esac
done

if [ "$DEV_MODE" = true ] && [ "$BUILD" = true ]; then
    DEV_MODE=true UPDATE_DEPS=$UPDATE_DEPS docker compose -f docker-compose-sciencegpt.yaml -f docker-compose-milvus.yaml up --pull always --build
elif [ "$DEV_MODE" = true ]; then
    DEV_MODE=true docker compose -f docker-compose-sciencegpt.yaml -f docker-compose-milvus.yaml up --pull always
elif [ "$BUILD" = true ]; then
    UPDATE_DEPS=$UPDATE_DEPS docker compose -f docker-compose-sciencegpt.yaml -f docker-compose-milvus.yaml up --pull always --build
else
    docker compose -f docker-compose-sciencegpt.yaml -f docker-compose-milvus.yaml up --pull always
fi
