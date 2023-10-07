import matplotlib.pyplot as plt

def plt_coils(image, plot_rows, plot_columns, image_index, n_channels=16, domain_selection=1, virtual=False):
    f, ax = plt.subplots(plot_rows, plot_columns, figsize=(32, 8))
    number_channels = 1
    for m in range(plot_rows):
        for n in range(plot_columns):
            ax[m][n].set_axis_off()
            if number_channels <= n_channels:
                # print("breaking | P: ")
                ax[m][n].imshow(abs(image[n + m * plot_columns, ...]), cmap='gray')
                number_channels = number_channels + 1

    if domain_selection == 1:
        domain = 'Image space'
        data_type = 'Diffusion Image'
    elif domain_selection == 2:
        domain = 'K-space'
        data_type = 'Diffusion Image'
    elif domain_selection == 3:
        domain = 'Sensitivity maps'
        data_type = 'Slice'
    else:
        domain = 'Domain was not selected'
        data_type = 'None'

    if virtual:
        channel_type = 'Virtual'
    else:
        channel_type = 'All'

    title = '{}: {} | {} Coils | {}'.format(data_type, image_index, virtual, domain)
    f.suptitle(title)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plt_contrasts(images, plot_rows, plot_columns, initial_contrast_image, n_contrasts,
                       condition='n + m * plot_columns', position_space=True):
    f, ax = plt.subplots(plot_rows, plot_columns, figsize=(8, 8))

    number_contrasts = 1
    if plot_rows == 1 and plot_columns == 1:
        ax.imshow(abs(images[...]), cmap='gray')
        ax.set_axis_off()
        caption = 'Contrast image [{}]'.format(initial_contrast_image)
    else:
        for m in range(plot_rows):
            for n in range(plot_columns):
                if number_contrasts > n_contrasts:
                    # print("breaking | ")
                    break
                # print(eval(condition))
                ax[m][n].set_axis_off()
                ax[m][n].imshow(abs(images[eval(condition), ...]), cmap='gray')
                number_contrasts = number_contrasts + 1

    if position_space:
        domain = 'Image space'
    else:
        domain = 'K-space'

    title = 'Contrasts images| {}'.format(domain)
    f.suptitle(title, fontsize=14)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plt_contrasts_02(images, plot_rows, plot_columns, array_contrast_image, n_contrasts, additional_caption,
                          position_space=True):
    # f, ax = plt.subplots(plot_rows, plot_columns, figsize=(8, 8))
    f, ax = plt.subplots(plot_rows, plot_columns, figsize=(8, 8))

    number_contrasts = 1
    if plot_rows == 1 and plot_columns == 1:
        ax.imshow(abs(images), cmap='gray')
        ax.set_axis_off()
        caption = 'Contrast image {} | '.format(str(array_contrast_image))
    else:
        caption = 'Contrast images | '
        for m in range(plot_rows):
            for n in range(plot_columns):
                if number_contrasts > n_contrasts:
                    # print("breaking | ")
                    break
                # print(eval(condition))
                ax[m][n].set_axis_off()
                ax[m][n].imshow(abs(images[array_contrast_image[number_contrasts - 1], ...]), cmap='gray')
                number_contrasts = number_contrasts + 1

    if position_space:
        domain = 'Image space'
    else:
        domain = 'K-space'

    title = caption + domain + additional_caption
    f.suptitle(title, fontsize=16, y=0.94)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
