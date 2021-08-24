def report():

    fig = plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of dataset', fontsize=24)
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    with open(str(output / "quality_report_features.html"), "a") as text_file:
        html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        text_file.write('<header><h1>UMAP Feature reduction </h1></header>')
        text_file.write(html)
        text_file.close()