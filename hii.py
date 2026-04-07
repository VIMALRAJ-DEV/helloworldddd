from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
corpus = ["The car is driven on the road", "The truck is driven on the highway"]
v = TfidfVectorizer(stop_words='english')
X = v.fit_transform(corpus)
print(pd.DataFrame(X.toarray(), columns=v.get_feature_names_out()))

import facebook 
access_token = 'TOKEN'
graph = facebook.GraphAPI(access_token) 
user_info = graph.get_object('1941812063362459')
print("User Info:")
print(user_info) 

import requests
import pandas as pd
import matplotlib.pyplot as plt

def get_user_repositories(username):
    url = f'https://api.github.com/users/{username}/repos'
   
    response = requests.get(url)
    return response.json()


repositories = get_user_repositories('VIMALRAJ-DEV')

df = pd.DataFrame(repositories)

plt.bar(df['name'], df['stargazers_count'])
plt.xlabel('Repository Name')
plt.ylabel('Number of Stars')
plt.title('GitHub Repository Stars')
plt.xticks(rotation=45, ha='right')
plt.show()




import requests

def get_user_info(username):
    url = f"https://api.github.com/users/{username}"
    
    
    response = requests.get(url)

    if response.status_code == 200:
        user_data = response.json()
        print(f"User: {user_data['login']}")
        print(f"Name: {user_data['name']}")
        print(f"Followers: {user_data['followers']}")
        print(f"Public Repositories: {user_data['public_repos']}")
    else:
        print(f"Error: {response.status_code}")

# Replace with real values
get_user_info("VIMALRAJ-DEV")