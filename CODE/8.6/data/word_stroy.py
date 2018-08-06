word_500 = open('word_500.txt', 'r')
story_500 = open('story_500.txt', 'r')

all_lines_word = word_500.readlines()
all_lines_story = story_500.readlines()

df_new = open('story_word_500.txt', 'w')
for line in zip(all_lines_word, all_lines_story):
    str_new = line[0][:-1] + "\t" + line[1]
    print(str_new)
    df_new.write(str_new)