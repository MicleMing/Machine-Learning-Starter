
import numpy as np
import matplotlib.pyplot as plt


def display_data(x):
    (m, n) = x.shape

    example_width = np.round(np.sqrt(n)).astype(int)
    example_height = np.round(n / example_width).astype(int)

    rows = np.floor(np.sqrt(m)).astype(int)
    cols = np.ceil(m / rows).astype(int)

    pad = 1

    display_array = -np.ones((
        pad + (example_height + pad) * rows,
        pad + (example_width + pad) * cols
    ))

    example_index = 0
    for i in range(rows):
        for j in range(cols):
            if example_index > m:
                break
            example_matrix = x[example_index].reshape(
                example_height, example_width
            )
            display_array[
                pad + i * (example_height + pad) + np.arange(example_height),
                pad + j * (example_width + pad) +
                np.arange(example_width)[:, np.newaxis]
            ] = example_matrix
            example_index += 1

        if example_index > m:
            break

    plt.figure()
    plt.imshow(display_array, cmap='gray', extent=[-1, 1, -1, 1])
    plt.axis('off')
