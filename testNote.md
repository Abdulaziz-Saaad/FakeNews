

# Prediction 0 (false)
```sh
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"statement": "The sky is blue."}'
```

# Prediction 1 (true)
```sh
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"statement": "The Biden administration is taking new measures aimed at stopping China from helping the Kremlin sustain its war effort against Ukraine. U.S. officials hope European nations will take similar steps."}'
```



### Real News Statements
1. "NASA’s Perseverance rover has successfully collected its first samples from the surface of Mars."
2. "The World Health Organization has declared the COVID-19 pandemic to be under control due to widespread vaccination efforts."
3. "The European Union has reached a historic agreement on climate change, committing to carbon neutrality by 2050."
4. "Apple has announced the release of its latest iPhone model, featuring advanced AI capabilities and a new design."
5. "Researchers at MIT have developed a new AI algorithm that can predict climate patterns with unprecedented accuracy."

### Fake News Statements
1. "Scientists have discovered a hidden city on the dark side of the moon, populated by ancient aliens."
2. "Drinking bleach can cure COVID-19, according to a viral social media post."
3. "The government is secretly replacing citizens' mobile phones with devices that track their every move."
4. "A new miracle drug promises to cure all types of cancer within 24 hours of a single dose."
5. "Elon Musk plans to implant microchips in human brains to control thoughts and actions remotely."