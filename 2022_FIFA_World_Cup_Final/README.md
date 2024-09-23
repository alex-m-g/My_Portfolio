# 2022 FIFA World Cup Final - Argentina vs. France

## Progres:
### Dataset Development
- Currently developing the dataset by screenshotting each penalty kick as the shooter plants the final foot to the ground.
- Gather all the screenshots and normalize them by cropping out the on-screen texts and referees, then resizing all images to 500x500 pixels.
- Unfortunately, as the camera shots for each penalty shootout are panned slightly differently for each penalty kick, I need to manually tweak the cropping for each image instead of iterating through an image folder for normalization.
