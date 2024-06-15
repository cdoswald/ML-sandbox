# Config params
$image_name = "waymo-od-image"
$container_name = "waymo-od-container"
$port = 8000

# Build docker image
docker build -t $image_name .

# Run docker container
docker run -it --name $container_name -p ${port}:${port} $image_name