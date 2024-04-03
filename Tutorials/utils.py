import matplotlib.pyplot as plt

def extract_complex_re_img(data):
    x = [ele.real for ele in data]
    y = [ele.imag for ele in data]
    return x, y

def plot_complex(data, title: str, figsize=(4,4)):
    fig = plt.figure(figsize=figsize)
    x, y = extract_complex_re_img(data)
    plt.scatter(x, y)
    ## Anotate
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        plt.text(x_i, y_i, f"$\\lambda_{i}$")
    plt.grid()
    plt.title(title)
    plt.xlabel("Real")
    plt.ylabel("Img")
    return fig