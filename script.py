import os
import shutil
import random
from pathlib import Path

def analyser_structure_projet():
    """
    Analyse la structure du projet OCT dans le répertoire courant
    """
    structure = {}
    
    # Parcourir les dossiers dans le répertoire courant
    for item in Path(".").iterdir():
        if item.is_dir() and item.name not in ['.git', '__pycache__', 'echantillons_oct']:
            structure[item.name] = []
            for sous_item in item.iterdir():
                if sous_item.is_dir():
                    structure[item.name].append(sous_item.name)
    
    return structure

def selectionner_echantillons(dossier_sortie="echantillons_oct"):
    """
    Sélectionne automatiquement des échantillons représentatifs
    """
    print("Démarrage de la sélection d'échantillons...")
    
    # Créer dossier de sortie
    if os.path.exists(dossier_sortie):
        shutil.rmtree(dossier_sortie)
    os.makedirs(dossier_sortie, exist_ok=True)
    
    # Vérifier la présence des dossiers
    dossiers_disponibles = [d for d in ["OD", "R_BIN1", "R_BIN2", "IMAGES"] 
                          if Path(d).exists()]
    
    print(f"Dossiers trouvés: {dossiers_disponibles}")
    
    # Stratégie de sélection adaptée
    if "OD" in dossiers_disponibles:
        print("Traitement du dossier OD...")
        traiter_dossier_od(dossier_sortie)
    
    if "R_BIN1" in dossiers_disponibles:
        print("Traitement du dossier R_BIN1...")
        traiter_dossier_rbin(Path("R_BIN1"), dossier_sortie, "R_BIN1")
    
    if "IMAGES" in dossiers_disponibles:
        print("Traitement du dossier IMAGES...")
        traiter_dossier_images(dossier_sortie)
    
    print(f"\nÉchantillons copiés dans: {dossier_sortie}")
    return dossier_sortie

def traiter_dossier_od(dossier_sortie):
    """
    Traite spécifiquement le dossier OD (RAD et CUBE)
    """
    chemin_od = Path("OD")
    
    for type_scan in ["RAD", "CUBE"]:
        chemin_type = chemin_od / type_scan
        if chemin_type.exists():
            copier_echantillons_images(chemin_type, 
                                     Path(dossier_sortie) / "OD" / type_scan, 
                                     nb_max=8)

def traiter_dossier_rbin(chemin_rbin, dossier_sortie, nom_groupe):
    """
    Traite les dossiers R_BIN1 ou R_BIN2
    """
    # Sélectionner quelques patients représentatifs
    patients_prioritaires = ["01_CAB_OD", "05_MIG_OD", "07_TIM_OD"]
    
    # Lister tous les patients disponibles
    patients_disponibles = [p.name for p in chemin_rbin.iterdir() if p.is_dir()]
    
    # Prendre les patients prioritaires s'ils existent, sinon les premiers
    patients_a_traiter = []
    for patient in patients_prioritaires:
        if patient in patients_disponibles:
            patients_a_traiter.append(patient)
    
    # Si pas assez, ajouter d'autres patients
    if len(patients_a_traiter) < 3:
        for patient in patients_disponibles:
            if patient not in patients_a_traiter:
                patients_a_traiter.append(patient)
                if len(patients_a_traiter) >= 3:
                    break
    
    # Traiter les patients sélectionnés
    for patient in patients_a_traiter:
        chemin_patient = chemin_rbin / patient
        if chemin_patient.exists():
            copier_echantillons_images(chemin_patient, 
                                     Path(dossier_sortie) / nom_groupe / patient, 
                                     nb_max=10)

def traiter_dossier_images(dossier_sortie):
    """
    Traite le dossier IMAGES (qui semble être une copie)
    """
    chemin_images = Path("IMAGES")
    
    # Prendre quelques sous-dossiers au hasard
    sous_dossiers = [d for d in chemin_images.iterdir() if d.is_dir()]
    
    if len(sous_dossiers) > 3:
        sous_dossiers_selection = random.sample(sous_dossiers, 3)
    else:
        sous_dossiers_selection = sous_dossiers
    
    for sous_dossier in sous_dossiers_selection:
        copier_echantillons_images(sous_dossier, 
                                 Path(dossier_sortie) / "IMAGES" / sous_dossier.name, 
                                 nb_max=5)

def copier_echantillons_images(source, destination, nb_max=5):
    """
    Copie un échantillon d'images depuis le dossier source
    """
    os.makedirs(destination, exist_ok=True)
    
    # Extensions d'images possibles pour l'OCT
    extensions_images = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.dcm', '.img'}
    
    # Lister tous les fichiers images
    fichiers_images = []
    for fichier in Path(source).iterdir():
        if fichier.is_file() and fichier.suffix.lower() in extensions_images:
            fichiers_images.append(fichier)
    
    if not fichiers_images:
        print(f"    Aucune image trouvée dans {source}")
        return
    
    # Sélectionner un échantillon
    if len(fichiers_images) > nb_max:
        # Prendre des images espacées régulièrement pour couvrir toute la série
        interval = len(fichiers_images) // nb_max
        if interval < 1:
            interval = 1
        fichiers_selectionnes = fichiers_images[::interval][:nb_max]
    else:
        fichiers_selectionnes = fichiers_images
    
    # Copier les fichiers sélectionnés
    for fichier in fichiers_selectionnes:
        try:
            shutil.copy2(fichier, destination / fichier.name)
        except Exception as e:
            print(f"    Erreur lors de la copie de {fichier.name}: {e}")
    
    print(f"    ✓ Copié {len(fichiers_selectionnes)}/{len(fichiers_images)} images de {source.name}")

def generer_rapport_echantillonnage():
    """
    Génère un rapport détaillé de la structure des données
    """
    print("\n" + "="*60)
    print("RAPPORT D'ANALYSE DU DATASET OCT")
    print("="*60)
    
    structure = analyser_structure_projet()
    
    for groupe, contenus in structure.items():
        print(f"\n📁 Groupe: {groupe}")
        print(f"   Nombre de sous-dossiers: {len(contenus)}")
        if contenus:
            print(f"   Contenus: {', '.join(contenus[:5])}")
            if len(contenus) > 5:
                print(f"   ... et {len(contenus)-5} autres")
    
    print("\n" + "-"*60)
    print("ANALYSE DES NOMENCLATURES")
    print("-"*60)
    print("• OD/OS: Œil Droit/Œil Gauche")
    print("• RAD: Images radiales (★ IMPORTANT pour votre projet)")
    print("• CUBE: Images en cube (volume 3D)")
    print("• Codes patients: CAB, TRK, TIM, MIG, BIM, DEA")
    print("• Numéros: 01-08 (identifiants patients)")
    
    print("\n" + "-"*60)
    print("RECOMMANDATIONS")
    print("-"*60)
    print("1. 🎯 FOCUS: Dossier OD/RAD pour modélisation fovéolaire")
    print("2. 🧪 TESTS: Patients de R_BIN1 pour développement")
    print("3. ✅ VALIDATION: R_BIN2 comme set de test final")
    print("4. 📊 IMAGES: Backup/exemples pour référence")

# Exécution principale
if __name__ == "__main__":
    print("🚀 SCRIPT D'ÉCHANTILLONNAGE OCT - DÉMARRAGE")
    print("-"*60)
    
    # Vérifier qu'on est dans le bon dossier
    fichiers_projet = [f for f in os.listdir(".") 
                      if f.startswith("Vision par ordinateur")]
    if fichiers_projet:
        print(f"✓ Projet détecté: {fichiers_projet[0]}")
    
    # Générer le rapport d'analyse
    generer_rapport_echantillonnage()
    
    # Créer les échantillons
    print("\n🔄 CRÉATION DES ÉCHANTILLONS...")
    dossier_echantillons = selectionner_echantillons()
    
    # Résumé final
    print("\n" + "="*60)
    print("✅ ÉCHANTILLONNAGE TERMINÉ!")
    print("="*60)
    print(f"📂 Dossier créé: {dossier_echantillons}")
    print("\n📋 PROCHAINES ÉTAPES:")
    print("1. Vérifiez le contenu du dossier 'echantillons_oct'")
    print("2. Zippez ce dossier pour l'envoyer")
    print("3. Ou envoyez quelques images individuelles")
    print("4. Nous commencerons le développement des algorithmes!")
    print("="*60)