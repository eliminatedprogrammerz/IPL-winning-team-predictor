import joblib
import warnings
warnings.filterwarnings("ignore")

model = joblib.load('model.pkl')
encoder_t1 = joblib.load('encoder_t1.pkl')
encoder_t2 = joblib.load('encoder_t2.pkl')
encoder_toss = joblib.load('encoder_toss.pkl')
encoder_toss_winner = joblib.load('encoder_toss_winner.pkl')
encoder_venue = joblib.load('encoder_venue.pkl')
encoder_winner = joblib.load('encoder_winner.pkl')

t1 = input("Enter the name of team 1: ")
t2 = input("Enter the name of team 2: ")
toss = input("Enter the name of the team that won the toss: ")
toss_decision = input("Enter the decision of the toss winner (bat/field): ")
venue = input("Enter the name of the venue: ")

t1_encoded = encoder_t1.transform([t1])
t2_encoded = encoder_t2.transform([t2])
toss_encoded = encoder_toss.transform([toss_decision])
toss_winner_encoded = encoder_toss_winner.transform([toss])
venue_encoded = encoder_venue.transform([venue])

prediction = model.predict([[
    t1_encoded[0], 
    t2_encoded[0], 
    toss_encoded[0], 
    toss_winner_encoded[0], 
    venue_encoded[0]
]])

winner = encoder_winner.inverse_transform(prediction)[0]

print(f"The predicted winner is: {winner}")