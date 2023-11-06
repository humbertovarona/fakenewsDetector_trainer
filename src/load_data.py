def load_data(filename):
    data = pd.read_csv(filename)
    data = data.sample(frac=1)
    return data
