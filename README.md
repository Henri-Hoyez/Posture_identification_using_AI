# Reconaissance de posture, d'allures et d'actions dans les vidéos.

![](demo.gif)

### Résumé du projet.

  Ces dernières années, le domaine de la reconnaissance de personnes par le visage à beaucoup progressé. Cependant, ses techniques restent innefficaces pour de grandes distances.
  La reconnaissance de personnes par l'étude de l'allure fait encore aujourd'hui l'objet de recherches. Ce projet organisé par l'école d'ingénieurs [ISEN-Lille](https://www.isen-lille.fr/) nous a donné l'occasion de découvrir ce domaine.
  
  Ce projet à été réalisé au sein d'un laboratoire reconstituant une maison connecté à [Urbawood](http://cs.isen.fr/) 

  L’objectif du projet est de reconnaître la position d’une personne ou une action à partir d’un flux vidéo, puis de reconnaître une personne uniquement par sa démarche.
  
  Vous pouvez voir une vidéo de démonstration en cliquant sur [ce lien](https://www.youtube.com/watch?v=SJaxPC9o-Vw&feature=youtu.be&fbclid=IwAR0VCNfuW4oD7jn_v5fOglbWuHDg_69B8UxLmtQigto5cMdVRRo8LYQDSyY).
  
  Nous avons fait le choix de diviser ce projet en 3 parties distinctes:

* [Pose](https://github.com/Steamyfuury/Posture_identification_using_AI/tree/master/Postures) - L'étude d'une posture statique,
* [Action](https://github.com/Steamyfuury/Posture_identification_using_AI/tree/master/Actions) - étude d'une action avec l'ajout de la dimension temporelle,
* [Allure](https://github.com/Steamyfuury/Posture_identification_using_AI/tree/master/Allures) - L'étude de la démarche d'une personne.

  Nous avons aussi fait le choix de mettre à disposition une [application de démonstration](https://github.com/Steamyfuury/Posture_identification_using_AI/tree/master/Demo).
  
 ### Fonctionnalitée  
 
  Notre projet dispose de nombreuses fonctionnalitée, tels que la reconnaissance de postures statiques, la reconnaissance d'actions et la reconnaissance d'allures.
  Nous avons décidé d'allé plus loins avec la reconnaissance d'action. En effet, nous avons également mis à disposition une classe [Requette.py](https://github.com/Steamyfuury/Posture_identification_using_AI/blob/master/Actions/requete.py) qui permet la manipulation d'object connecté via requette HTTP. 

 ### Installation   
  
  Pour ce projet, nous utilisons un framework de vision artificielle appelée [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) qui nous permet d'extraire le squelette d'une personne à partir d'une image. Il est nécessaire de [l'installer](https://github.com/CMU-Perceptual-Computing-Lab/openpose#installation-reinstallation-and-uninstallation) avant d'utiliser notre projet.
  Une fois l'installation terminée, déposez les dossiers du projet dans:
  
```
  build/examples/tutorial_api_python
```
  Une fois les dossiers copiés vous pouvez, normalement utiliser le projet.
  

  

