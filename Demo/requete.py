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
	elif(id==4):
		suffixeUrl="bravia/power/on"
	elif(id==5):
		suffixeUrl="bravia/power/off"
	url=radicalUrl+suffixeUrl
	print(url)
	request.urlopen(url).read()
	
if __name__ == "__main__":
	requete(4)