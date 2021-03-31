
# Define the architecture by modifying resnet.
# Original code is here http://tiny.cc/8zpmmz
class ResNet101(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=True, **kwargs):
        # Start with the standard resnet101
        super().__init__(
            block=models.resnet.Bottleneck,
            layers=[3, 4, 23, 3],
            num_classes=num_classes,
            **kwargs
        )
        if pretrained:
            state_dict = load_state_dict_from_url(
                models.resnet.model_urls['resnet101'],
                progress=True
            )
            self.load_state_dict(state_dict)

    # Reimplementing forward pass.
    # Replacing the forward inference defined here
    # http://tiny.cc/23pmmz
    def _forward_impl(self, x):
        # Standard forward for resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Notice there is no forward pass through the original classifier.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

# initialize our implementation of ResNet
model = ResNet101(pretrained=True)
model.eval()

for batch in tqdm(dataloader, desc='Running the model inference'):
    images = batch['image'].to(device)
    labels += batch['label']
    image_paths += batch['image_path']

    output = model.forward(images)

    current_outputs = output.cpu().numpy()
    features = np.concatenate((outputs, current_outputs))


tsne = TSNE(n_components=2).fit_transform(features)

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)


# initialize a matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111)

# for every class, we'll add a scatter plot separately
for label in colors_per_class:
    # find the samples of the current class in the data
    indices = [i for i, l in enumerate(labels) if l == label]

    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)

    # convert the class color to matplotlib format
    color = np.array(colors_per_class[label], dtype=np.float) / 255

    # add a scatter plot with the corresponding color and label
    ax.scatter(current_tx, current_ty, c=color, label=label)

# build a legend using the labels we set previously
ax.legend(loc='best')

# finally, show the plot
plt.show()


# Compute the coordinates of the image on the plot
def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center,
    # compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y

# we'll put the image centers in the central area of the plot
# and use offsets to make sure the images fit the plot

# init the plot as white canvas
tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

# now we'll put a small copy of every image to its corresponding T-SNE coordinate
for image_path, label, x, y in tqdm(
        zip(images, labels, tx, ty),
        desc='Building the T-SNE plot',
        total=len(images)
):
    image = cv2.imread(image_path)

    # scale the image to put it to the plot
    image = scale_image(image, max_image_size)

    # draw a rectangle with a color corresponding to the image class
    image = draw_rectangle_by_class(image, label)

    # compute the coordinates of the image on the scaled plot visualization
    tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

    # put the image to its t-SNE coordinates using numpy sub-array indices
    tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

cv2.imshow('t-SNE', tsne_plot)
cv2.waitKey()

print('Funciona?')