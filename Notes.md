## Contexte

Nous réalisons un POC avec l'association (bien réelle) du prof, qui devrait pouvoir nous fournir des données sur lesquelles développer



# Sujets

### Obtention des données
- Un dossier par an, le client benne tout?

### Preprocessing
- Vérifier la taille des données

- Transformation des formats de données


## Créneau marché
### Concurrents
- ERP => Vise des structures plus grosses ?
- Outils de rédaction classiques => Lents, manuels et besoin de retrouver les données

### Avantage compétitif

RAGnar se distingue des autres solutions pour les associations par sa simplicité d'utilisation et sa capacité à s'adapter aux environnements peu structurés, typiques des petites associations ou de celles en début de développement. Contrairement à des outils comme Salesforce ou QuickBooks, qui peuvent être complexes et nécessiter une courbe d'apprentissage importante, RAGnar est conçu pour être intuitif, permettant aux utilisateurs sans expertise technique de générer facilement des rapports annuels. Il offre une interface épurée et des fonctionnalités adaptées aux besoins spécifiques des associations, telles que des modèles de rapports personnalisables, une intégration facile des données financières et des outils de collaboration. Cette simplicité garantit que même les associations avec peu de ressources peuvent se concentrer sur leur mission, sans se perdre dans des outils trop sophistiqués ou coûteux


# Fonctionalités

## Pour le POC

Pour un proof of concept (POC), les objectifs doivent être réalistes, avec un focus sur le cœur de la proposition de valeur, tout en minimisant les fonctionnalités secondaires. Voici une version simplifiée et adaptée :

### 1. **Interface minimale et fonctionnelle**
   - Créer une interface simple, intuitive, et testable, sans nécessiter de design avancé.
   - Fournir un tutoriel basique pour guider les utilisateurs à travers le processus.

### 2. **Création de rapports basique**
   - Offrir un modèle de rapport préconfiguré (par exemple, un rapport financier simple ou un résumé d’activités).  
   - Permettre à l'utilisateur d'importer des données sous forme de tableau (CSV ou Excel).

### 3. **Exportation des rapports**
   - Intégrer une fonctionnalité de base pour exporter le rapport au format PDF ou Word.  
   - Assurer que le format exporté est lisible et fonctionnel.

### 4. **Génération de visuels simplifiés**
   - Inclure un outil basique pour générer un graphique ou un tableau à partir des données importées (par exemple, un diagramme à barres ou un camembert).  

### 5. **Collaboration rudimentaire**
   - Permettre l'accès multi-utilisateurs à un projet via un partage de lien ou un fichier exportable.  

### 6. **Focus sur la simplicité**
   - Limiter les intégrations complexes avec d'autres outils (comme des bases de données externes ou des API).  
   - Tester l'ensemble sur des scénarios réels mais simples, typiques des associations ciblées.

### Priorités ajustées :
- **Expérience utilisateur :** La navigation doit être intuitive, même si l'apparence est basique.  
- **Résultat concret :** Le rapport généré doit démontrer la faisabilité de la solution et le potentiel d'amélioration future.  
- **Feedback utilisateur :** Prévoir une méthode pour recueillir les retours des utilisateurs pour affiner la version suivante.

Avec ces éléments, votre POC pourra valider les concepts clés, démontrer la valeur ajoutée de RAGnar et préparer une transition vers un prototype plus avancé.


## À terme

Pour que RAGnar soit compétitif dans le domaine de la rédaction de rapports annuels pour associations, voici le niveau minimal de fonctionnalités qu'il doit atteindre :

### 1. **Interface intuitive et accessible**  
   - Permettre à des utilisateurs non techniques de naviguer facilement dans l'application.  
   - Inclure des tutoriels intégrés ou une aide contextuelle pour guider les utilisateurs.

### 2. **Modèles de rapport personnalisables**  
   - Fournir des modèles préconfigurés pour des rapports annuels financiers, narratifs ou d’impact, tout en permettant des ajustements visuels et structurels selon les besoins des associations.

### 3. **Gestion et intégration des données**  
   - Intégrer facilement des données provenant de feuilles Excel, Google Sheets, ou autres outils courants utilisés par les associations.  
   - Centraliser les données financières, les statistiques d’activités, et les récits d’impact dans un format prêt à être utilisé.

### 4. **Collaboration simplifiée**  
   - Autoriser plusieurs utilisateurs à travailler simultanément sur un rapport, avec un suivi des modifications (comme Google Docs).  
   - Proposer des options de partage pour recueillir des avis et contributions.

### 5. **Génération automatique de graphiques et visuels**  
   - Inclure un outil intégré pour transformer les données en graphiques et diagrammes, sans nécessiter de compétences avancées en design.

### 6. **Accessibilité multi-format**  
   - Permettre l’exportation de rapports en PDF, Word, ou formats web interactifs.  
   - Optimiser les rapports pour les impressions ou les consultations numériques (infographies, vidéos intégrées, etc.).

### 7. **Adaptabilité aux contraintes des petites associations**  
   - Être abordable, avec un modèle freemium pour les petites structures.  
   - Nécessiter peu ou pas de configuration technique à l’installation.

### 8. **Sécurité et transparence des données**  
   - Garantir la sécurité des informations sensibles, comme les données financières et les informations sur les membres.  
   - Offrir une transparence sur les processus, essentielle pour les rapports financiers destinés aux donateurs.

### Justification :  
Ces fonctionnalités répondent directement aux lacunes identifiées dans les solutions existantes comme **Salesforce Nonprofit Cloud** (complexité) ou **Google Docs** (manque d’automatisation). L’objectif est de rendre RAGnar indispensable pour les associations recherchant simplicité et efficacité.



## Docs disponibles
- Rapports d'activité passés
- Rapports d'assemblée générale AGO-AGE / conseil d'admin
- Statuts de l'asso / règlement / charte
- Photos (metadata) -> pour illustrer
- Documents de compte
- Factures
- Fichiers Facebook / Insta / Réseaux sociaux
- Mails
- Affiches
- Discord
- Agenda
- Plans bâtiments / devis / études archi
- Quittances/ Convention de loyers
- Articles de presse
- Contentieux

## Prétraitement et transformations 
### **1. Rapports d'activité / AGO-AGE / Conseil d'administration**  
- **Récupération :** OCR (si fichiers scannés) via Tesseract ou un outil comme PyPDF2 pour extraire le texte.  
- **Transformation :** Analyse sémantique pour identifier les sections importantes (ex. résultats, projets réalisés). Utiliser spaCy ou GPT pour la synthèse automatique.  

---

### **2. Statuts, règlements, chartes**  
- **Récupération :** Extraction de texte (similaire aux rapports).  
- **Transformation :** Regrouper les extraits pertinents en utilisant des outils de recherche (Elasticsearch ou Whoosh) pour faciliter leur inclusion dans le rapport.  

---

### **3. Photos (métadonnées)**  
- **Récupération :** Extraire les métadonnées avec exiftool ou Pillow.  
- **Transformation :** Filtrer par date et lieu, créer des catégories associées à des événements ou projets (basées sur les métadonnées ou le titre du fichier).  

---

### **4. Documents de compte / Factures**  
- **Récupération :** Lire les fichiers Excel ou PDF avec pandas, PyPDF2 ou Tabula.  
- **Transformation :** Résumer les dépenses et recettes en visualisations simples avec Matplotlib ou Seaborn.  

---

### **5. Fichiers Facebook / Instagram / Réseaux sociaux**  
- **Récupération :** Si disponibles sous forme de fichiers (JSON ou CSV), parser directement. Sinon, scraper les données publiques avec un outil comme Apify (attention aux conditions d'utilisation).  
- **Transformation :** Résumer les statistiques (nombre de posts, engagements) et extraire des tendances avec NLP (NLTK ou spaCy).  

---

### **6. Mails**  
- **Récupération :** Importer les fichiers EML ou exporter les mails en JSON/CSV. Utiliser `imaplib` pour se connecter à une boîte mail.  
- **Transformation :** Filtrer par sujet et date, puis analyser les échanges pour extraire des informations clés (NLP pour classifier les mails par thèmes).  

---

### **7. Affiches / Articles de presse**  
- **Récupération :** OCR pour extraire le texte des affiches et des articles.  
- **Transformation :** Extraire des citations ou messages importants, catégoriser par événement ou date.  

---

### **8. Discord**  
- **Récupération :** Exporter l’historique via des outils comme DiscordChatExporter.  
- **Transformation :** NLP pour détecter les discussions pertinentes et résumer les échanges principaux (topics fréquents, actions décidées).  

---

### **9. Agenda**  
- **Récupération :** Importer les données d’agenda en CSV ou iCal.  
- **Transformation :** Générer une chronologie des événements clés pour le rapport avec Plotly ou Matplotlib.  

---

### **10. Plans / devis / études d’archi**  
- **Récupération :** Analyser les fichiers PDF ou CAD avec des outils spécifiques comme ezdxf pour les plans techniques.  
- **Transformation :** Résumer les informations financières ou techniques (dates, coûts, étapes).  

---

### **11. Quittances / Conventions de loyers**  
- **Récupération :** Extraction des données tabulaires (ex. montants, dates) avec Tabula ou Camelot pour les PDF.  
- **Transformation :** Produire un tableau résumé des loyers payés/à payer et générer des graphiques.  

---

### **12. Contentieux**  
- **Récupération :** OCR pour les documents papier et extraction de texte des fichiers numériques.  
- **Transformation :** Analyse sémantique pour résumer les cas principaux et leur statut.  
