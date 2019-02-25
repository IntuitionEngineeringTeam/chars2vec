import chars2vec
import sklearn.decomposition
import matplotlib.pyplot as plt


# load trained model of dimension 50
c2v_model = chars2vec.load_model('eng_50')

words = ['Natural', 'Language', 'Understanding',
         'Naturael', 'Longuge', 'Updderctundjing',
         'Motural', 'Lamnguoge', 'Understaating',
         'Naturrow', 'Laguage', 'Unddertandink',
         'Nattural', 'Languagge', 'Umderstoneding']

# create word embeddings
word_embeddings = c2v_model.vectorize_words(words)

# project embeddings on plane using the PCA
projection_2d = sklearn.decomposition.PCA(n_components=2).fit_transform(word_embeddings)

# draw words on plane
f = plt.figure(figsize=(8, 6))

for j in range(len(projection_2d)):
    plt.scatter(projection_2d[j, 0], projection_2d[j, 1],
                marker=('$' + words[j] + '$'),
                s=500 * len(words[j]), label=j,
                facecolors='green' if words[j] in ['Natural', 'Language', 'Understanding'] else 'black')

plt.show()
