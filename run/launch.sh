sudo docker container rm rawformer_exp
sudo docker build -t rawformer ../
sudo nvidia-docker run \
    --name rawformer_exp \
    --shm-size=50gb \
    -v /home/rst/dataset/ASV_spoof:/data \
    rawformer:latest