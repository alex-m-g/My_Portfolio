# 2022 FIFA World Cup Final - Argentina vs. France

## Progres:
### Dataset Development
- Currently developing the dataset by screenshotting each penalty kick as the shooter plants the final foot to the ground.
- Gathering all the screenshots and normalizing them by cropping out the on-screen texts and referrees; then resizing all images to 500x500 pixels.
- Unfortunately as the camera shots for each penalty shootout is panned slightly differently for each penalty kick, I need to manually tweak the cropping for each image instead of iterating through a image folder for normalization.
- Normalization File: [Normalization Tool](2022_FIFA_World_Cup_Final/image_normalization.py)
