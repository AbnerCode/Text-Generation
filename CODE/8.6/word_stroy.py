import pandas as pd

story = pd.read_csv('story_30.csv', header=-1)
word = pd.read_csv('word_30.csv', header=-1)

story['word'] = word.loc[:, 0]

story.to_csv('story_word.txt', sep='\t', index=False, header=False)