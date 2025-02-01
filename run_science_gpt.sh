mkdir ./app/data # dir must exist for dockerfile

DEV_MODE=false
BUILD=false
UPDATE_DEPS=false
NO_GPU=false

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
        --no-gpu)
        NO_GPU=true
        ;;
    esac
done

if [ "$DEV_MODE" = true ] && [ "$BUILD" = true ]; then
    GPU=$NO_GPU DEV_MODE=true UPDATE_DEPS=$UPDATE_DEPS docker compose -f docker-compose-sciencegpt.yaml -f docker-compose-milvus.yaml up --pull always --build
elif [ "$DEV_MODE" = true ]; then
    GPU=$NO_GPU DEV_MODE=true docker compose -f docker-compose-sciencegpt.yaml -f docker-compose-milvus.yaml up --pull always
elif [ "$BUILD" = true ]; then
    GPU=$NO_GPU UPDATE_DEPS=$UPDATE_DEPS docker compose -f docker-compose-sciencegpt.yaml -f docker-compose-milvus.yaml up --pull always --build
else
    GPU=$NO_GPU docker compose -f docker-compose-sciencegpt.yaml -f docker-compose-milvus.yaml up --pull always
fi
