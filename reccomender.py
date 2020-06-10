import pandas as pd
import numpy as np

# Creating array of existing data
sample_data = pd.read_csv("sample_data.csv")
ratings = np.array(sample_data.values)

# Calculating the average of existing data
avg = [0]
new = []
for i in range(1,len(ratings[0])):
    sum = 0
    for j in range(len(ratings)):
        sum += ratings[j][i]
    avg.append(sum / len(ratings))
for i in range(1,len(ratings[0])):
    temp = []
    for j in range(len(ratings)):
        temp.append(ratings[j][i] - avg[i])
    new.append(temp)
new = np.array(new).T
U,S,VT = np.linalg.svd(new)

# Truncating SVD to the three most significant values
r = S[:3]
U_truncated = U[:,:3]
S_truncated = np.diag(r)
VT_truncated = VT[:
                  3,:]
ratings_approx = U_truncated.dot(S_truncated.dot(VT_truncated))
error = np.linalg.norm(new - ratings_approx)/np.linalg.norm(new) * 100


# Getting names of movies
movies = list(sample_data.columns)[1:]
print()
print("Availible movies: ")
print(movies)
print()

# Asking user for which movies they have seen
watched = []
while True:
    curr_watched = input("Select a movie you have watched from the list " +
        "above and press enter. If there are none, type 'None' and press enter. ")
    if curr_watched == "None":
        if not watched:
            print("Please select at least one movie.")
        else:
            print()
            break
    else:
        if curr_watched not in movies:
            print("You have entered a movie which is not in the list above. Please select another one or enter None to finish.")
        elif curr_watched in watched:
            print("You have already selected that movie. Please select another one or None.")
        else:
            watched.append(curr_watched)
    print()

# Asking user to rate each movie from a scale of 1 to 5
# Movie and ratings are stored within dictionary
user = {}
for movie in watched:
    score = input("How would you rate " + movie + " on a scale from 1 to 5? ")
    while True:
        try:
            score = int(score)
            if score > 5 or score < 1:
                score = input("Please select a number from 1 to 5. ")
            else:
                print()
                break
        except:
            score = input("Please select a number from 1 to 5. ")
    user[movie] = score

# Processing input data to create vector representation
user_ratings = []
temp = []
for i in range(1,len(sample_data.columns)):
    if user.get(list(sample_data.columns)[i]) == None:
        temp.append(avg[i])
    else:
        temp.append(user.get(list(sample_data.columns)[i]))
user_ratings.append(temp)
for i in range(len(user_ratings)):
    for j in range(len(user_ratings[0])):
        user_ratings[i][j] = user_ratings[i][j] - avg[j + 1]

# Caclulating the input vector representation in terms of the truncated SVD
proj_user = np.zeros(25)
for i in range(len(VT_truncated)):
    proj_user = proj_user + (np.array(user_ratings[0]) @ VT_truncated[i].T) * VT_truncated[i]

# Adding mean back into preference vector to reaccount for bias
pred_user = proj_user + avg[1:]

# Outputting reccomendation
print("Top five reccomendations: ")
count = 1
while count < 6:
    curr_reccomendation = list(sample_data.columns)[np.argmax(pred_user) + 1]
    if curr_reccomendation in watched:
        pred_user[np.argmax(pred_user)] = 0
    else:
        print(str(count) + ". " + curr_reccomendation)
        pred_user[np.argmax(pred_user)] = 0
        count += 1
