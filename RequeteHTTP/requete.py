from urllib import request,parse

channel=3
def requete(id):
	global channel

	radicalUrl="http://10.34.168.135/api/"
	suffixeUrl=""

	if(id==0):
		suffixeUrl="allLights/on/0"
	
	elif(id==1):
		suffixeUrl="allLights/off/0"
	
	elif(id==2):
		channel+=1
		suffixeUrl="tv/channel/"+str(channel)
	
	elif(id==3):
		channel-=1
		suffixeUrl="tv/channel/"+str(channel)
	url=radicalUrl+suffixeUrl
	print(url)
	request.urlopen(url).read()

requete(0)