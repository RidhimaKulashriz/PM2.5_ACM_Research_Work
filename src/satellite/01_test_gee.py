import ee

ee.Authenticate()
ee.Initialize(project="delhi-pm25-research")

print(ee.String("Earth Engine working!").getInfo())