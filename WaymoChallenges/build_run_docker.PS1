# Config params
$image_name = "waymo-od-image"
$container_name = "waymo-od-container"
$volume_local_path = "C:/Users/chris/GitHub/ML-sandbox/WaymoChallenges"
$volume_contain_path = "/workspace"
$port = 8000

# Build docker image
docker build -t $image_name .

# Run docker container
docker run -v ${volume_local_path}:${volume_contain_path} -it --name $container_name -p ${port}:${port} $image_name