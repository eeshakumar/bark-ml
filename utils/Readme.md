# Building and Uploading Images

## Docker
`docker login`
`docker build -t barksim/bark-ml .`
`docker push barksim/bark-ml:latest`
`docker prune -a`

## Singularity
`sudo singularity build bark_ml.img Singularity`
`sbash run.sh`
`bash upload_image.sh hart`

## Cluster
Mount drive:
`sudo mount -t glusterfs -o acl fortiss-8gpu:/data /mnt/glusterdata`