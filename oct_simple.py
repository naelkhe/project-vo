import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
import json
import datetime

# Dossier de résultats
OUTPUT_DIR = "resultats_oct_simple"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyse_oct_images(image_paths):
    """
    Fonction principale d'analyse des images OCT
    """
    print(f"Analyse de {len(image_paths)} images OCT...")
    
    results = []
    ilm_curves = []
    angles = []
    
    for i, img_path in enumerate(image_paths):
        print(f"\nTraitement de l'image {i+1}: {Path(img_path).name}")
        
        # Charger l'image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️ Impossible de charger l'image {img_path}")
            continue
        
        # Extraire l'angle de l'image
        angle = extract_angle(img_path, i)
        
        # Extraire la partie OCT (coupe)
        oct_section = extract_oct_section(image)
        
        # Détecter l'ILM
        ilm_x, ilm_y = detect_ilm(oct_section)
        
        if ilm_x is not None and ilm_y is not None:
            # Sauvegarder la courbe ILM
            ilm_curves.append({
                'x': ilm_x,
                'y': ilm_y,
                'angle': angle
            })
            angles.append(angle)
            
            # Visualiser et sauvegarder l'ILM détectée
            save_ilm_visualization(oct_section, ilm_x, ilm_y, angle, i)
            
            print(f"✓ ILM détectée pour l'angle {angle}°")
        else:
            print(f"⚠️ Échec de la détection ILM pour l'image {i+1}")
    
    # Si des courbes ILM ont été détectées
    if ilm_curves:
        # Reconstruire la surface 3D
        surface_3d = reconstruct_3d_surface(ilm_curves)
        
        # Ajuster un modèle gaussien
        model_params = fit_gaussian_model(surface_3d)
        
        # Visualiser les résultats
        if surface_3d and model_params:
            visualize_3d_surface(surface_3d)
            visualize_gaussian_fit(surface_3d, model_params)
            
            # Sauvegarder les résultats en JSON
            save_results(angles, model_params, surface_3d)
            
            return True
    
    print("⚠️ Analyse incomplète")
    return False

def extract_angle(filename, index):
    """
    Extrait l'angle depuis le nom de fichier
    """
    # Format attendu "CONTROL_CHC_20241022_RAD_OD_XXX.tif"
    import re
    filename_str = str(filename)
    
    # Rechercher divers formats d'angle
    rad_match = re.search(r'RAD_OD_(\d+)', filename_str)
    if rad_match:
        file_index = int(rad_match.group(1))
        angle = int(file_index * (360 / 28))  # Répartir les angles sur 360°
        return angle
    
    q_match = re.search(r'Q:\s*(\d+)', filename_str)
    if q_match:
        return int(q_match.group(1))
    
    angle_match = re.search(r'(\d+)[°\s]', filename_str)
    if angle_match:
        return int(angle_match.group(1))
    
    # Angle par défaut basé sur l'index
    default_angles = [0, 12, 25, 38, 51, 64, 77, 90, 103, 116, 
                      129, 142, 155, 168, 180, 193, 206, 219, 
                      232, 245, 258, 271, 284, 297, 310, 323, 336, 348]
    
    if index < len(default_angles):
        return default_angles[index]
    
    return index * 15  # Angles de 15° en 15°

def extract_oct_section(image):
    """
    Extrait la section OCT de l'image composite
    """
    height, width = image.shape[:2]
    
    # Si l'image est suffisamment large pour être composite
    if width > height * 1.5:
        # Tenter de trouver la séparation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vertical_variance = np.var(gray, axis=0)
        
        # Chercher la séparation dans la moitié centrale
        search_start = width // 4
        search_end = 3 * width // 4
        middle_section = vertical_variance[search_start:search_end]
        
        # Trouver le point de variance minimale
        separation_idx = np.argmin(middle_section) + search_start
        
        # Partie droite = OCT
        oct_section = image[:, separation_idx:]
    else:
        # Si l'image n'est pas composite, prendre l'image entière
        oct_section = image
    
    return oct_section

def detect_ilm(oct_image):
    """
    Détecte la limitante interne (ILM) dans l'image OCT
    Version améliorée qui suit bien la surface de la rétine
    """
    # Conversion en niveaux de gris
    if len(oct_image.shape) == 3:
        gray = cv2.cvtColor(oct_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = oct_image.copy()
    
    height, width = gray.shape
    
    # 1. Amélioration du contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 2. Réduction du bruit (préservant les bords)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 3. Détection des bords avec gradient vertical
    sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    
    # 4. Détection de la rétine (transition noir -> blanc)
    ilm_points = []
    
    # 5. Définir la région de recherche (éviter le noir en haut)
    roi_top = int(height * 0.2)      # Ignorer les 20% supérieurs
    roi_bottom = int(height * 0.5)   # Chercher dans la moitié supérieure
    
    # 6. Extraire les points ILM
    for x in range(width):
        column = sobel_y[roi_top:roi_bottom, x]
        
        # Chercher les transitions positives fortes (noir -> blanc)
        if column.size > 0:
            # Trouver le point avec le gradient le plus fort
            max_y = np.argmax(column)
            actual_y = roi_top + max_y
            
            # Vérifier qu'il s'agit d'une transition significative
            if column[max_y] > np.std(column) * 2.5:
                ilm_points.append((x, actual_y))
    
    # 7. Vérifier s'il y a assez de points
    if len(ilm_points) < width // 5:  # Au moins 20% de la largeur
        print(f"Détection ILM faible: seulement {len(ilm_points)} points trouvés")
        return None, None
    
    # 8. Convertir en arrays numpy
    points = np.array(ilm_points)
    x_points = points[:, 0]
    y_points = points[:, 1]
    
    # 9. Trier par position X
    sort_indices = np.argsort(x_points)
    x_sorted = x_points[sort_indices]
    y_sorted = y_points[sort_indices]
    
    # 10. Filtrer les valeurs aberrantes
    window_size = min(9, len(y_sorted)//2)
    if window_size > 2:
        y_median = cv2.medianBlur(y_sorted.astype(np.float32).reshape(-1, 1), window_size).reshape(-1)
    else:
        y_median = y_sorted
    
    # 11. Interpolation lisse sur toute la largeur
    try:
        # Ajustement polynomial pour obtenir une courbe lisse
        # Utiliser un degré variable selon le nombre de points
        degree = min(4, len(x_sorted) // 50)
        poly_coeffs = np.polyfit(x_sorted, y_median, degree)
        
        # Générer une courbe lisse sur toute la largeur
        x_smooth = np.arange(width)
        y_smooth = np.polyval(poly_coeffs, x_smooth)
        
        return x_smooth, y_smooth
    except:
        # En cas d'échec, essayer une interpolation simple
        x_smooth = np.arange(width)
        y_smooth = np.interp(x_smooth, x_sorted, y_median)
        return x_smooth, y_smooth

def save_ilm_visualization(oct_image, ilm_x, ilm_y, angle, index):
    """
    Sauvegarde une visualisation de l'ILM détectée
    """
    # Convertir en RGB pour l'affichage
    if len(oct_image.shape) == 3:
        display_img = cv2.cvtColor(oct_image, cv2.COLOR_BGR2RGB)
    else:
        display_img = cv2.cvtColor(oct_image, cv2.COLOR_GRAY2RGB)
    
    # Créer une copie pour dessiner dessus
    img_with_ilm = display_img.copy()
    
    # Dessiner l'ILM en rouge
    for i in range(len(ilm_x)-1):
        cv2.line(img_with_ilm, 
                (int(ilm_x[i]), int(ilm_y[i])), 
                (int(ilm_x[i+1]), int(ilm_y[i+1])), 
                (255, 0, 0), 2)
    
    # Ajouter le texte d'angle
    cv2.putText(img_with_ilm, f"Angle: {angle}°", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Sauvegarder l'image
    plt.figure(figsize=(10, 6))
    plt.imshow(img_with_ilm)
    plt.axis('off')
    plt.title(f'Détection ILM - Angle {angle}°')
    
    plt.savefig(f"{OUTPUT_DIR}/ilm_{index:02d}_{angle}.png", 
               dpi=150, bbox_inches='tight')
    plt.close()

def reconstruct_3d_surface(ilm_curves):
    """
    Reconstruit une surface 3D à partir des courbes ILM
    """
    if len(ilm_curves) < 1:
        return None
    
    # Convertir en coordonnées polaires
    polar_data = []
    
    for curve in ilm_curves:
        # Angle en radians
        angle_rad = np.deg2rad(curve['angle'])
        
        # Trouver le centre de la dépression
        min_idx = np.argmin(curve['y'])
        center_x = curve['x'][min_idx]
        center_y = curve['y'][min_idx]
        
        # Coordonnées relatives au centre
        x_rel = curve['x'] - center_x
        y_rel = curve['y'] - center_y
        
        # Conversion en coordonnées polaires
        r = np.abs(x_rel)
        
        # Assurer des valeurs uniques de r
        r_unique, unique_indices = np.unique(r, return_index=True)
        y_unique = y_rel[unique_indices]
        
        polar_data.append({
            'angle': angle_rad,
            'r': r_unique,
            'z': y_unique,
            'center': (center_x, center_y)
        })
    
    # Grille polaire uniforme
    angles = np.linspace(0, 2*np.pi, 72)  # 72 angles (5° par pas)
    max_radius = max([np.max(data['r']) for data in polar_data])
    radii = np.linspace(0, max_radius, 100)
    
    # Créer la surface 3D
    surface = np.zeros((len(angles), len(radii)))
    
    # Pour chaque angle de la grille
    for i, angle in enumerate(angles):
        # Trouver la courbe la plus proche
        closest_idx = 0
        min_diff = 2*np.pi
        
        for j, data in enumerate(polar_data):
            # Différence d'angle (gestion des valeurs circulaires)
            diff = np.abs(np.mod(angle - data['angle'] + np.pi, 2*np.pi) - np.pi)
            if diff < min_diff:
                min_diff = diff
                closest_idx = j
        
        # Utiliser les données les plus proches
        closest_data = polar_data[closest_idx]
        
        # Interpolation
        try:
            from scipy.interpolate import interp1d
            # Utiliser interpolation linéaire (plus robuste)
            interp = interp1d(closest_data['r'], closest_data['z'], 
                            kind='linear', bounds_error=False, fill_value=0)
            surface[i, :] = interp(radii)
        except:
            # Fallback avec numpy
            surface[i, :] = np.interp(radii, closest_data['r'], closest_data['z'], 
                                    left=0, right=0)
    
    return {
        'surface': surface,
        'angles': angles,
        'radii': radii
    }

def fit_gaussian_model(surface_data):
    """
    Ajuste un modèle gaussien à la surface
    """
    if not surface_data:
        return None
    
    # Extraire les données
    surface = surface_data['surface']
    radii = surface_data['radii']
    
    # Calculer le profil radial moyen
    radial_profile = np.nanmean(surface, axis=0)
    
    # Remplacer les NaN par des zéros
    radial_profile = np.nan_to_num(radial_profile)
    
    # Fonction du modèle gaussien
    def gaussian(r, A, sigma, z0):
        return A * np.exp(-(r**2) / (2 * sigma**2)) + z0
    
    try:
        from scipy.optimize import curve_fit
        
        # Estimation initiale
        A_init = np.min(radial_profile) - np.max(radial_profile)
        sigma_init = radii[len(radii)//4]
        z0_init = np.max(radial_profile)
        
        # Ajustement
        popt, pcov = curve_fit(gaussian, radii, radial_profile,
                              p0=[A_init, sigma_init, z0_init])
        
        A, sigma, z0 = popt
        
        # Qualité de l'ajustement
        y_pred = gaussian(radii, *popt)
        ss_total = np.sum((radial_profile - np.mean(radial_profile))**2)
        ss_residual = np.sum((radial_profile - y_pred)**2)
        
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        rmse = np.sqrt(np.mean((radial_profile - y_pred)**2))
        
        return {
            'type': 'gaussian',
            'amplitude': float(A),
            'sigma': float(sigma),
            'baseline': float(z0),
            'r_squared': float(r_squared),
            'rmse': float(rmse),
            'y_pred': y_pred
        }
    except Exception as e:
        print(f"Erreur lors de l'ajustement du modèle: {e}")
        return None

def visualize_3d_surface(surface_data):
    """
    Visualise la surface 3D
    """
    if not surface_data:
        return
    
    try:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Conversion en coordonnées cartésiennes
        R, THETA = np.meshgrid(surface_data['radii'], surface_data['angles'])
        X = R * np.cos(THETA)
        Y = R * np.sin(THETA)
        Z = surface_data['surface']
        
        # Surface 3D
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel('Profondeur (pixels)')
        ax.set_title('Surface 3D de la Dépression Fovéolaire')
        
        # Barre de couleur
        fig.colorbar(surf)
        
        plt.savefig(f"{OUTPUT_DIR}/surface_3d.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Surface 3D sauvegardée: {OUTPUT_DIR}/surface_3d.png")
    except Exception as e:
        print(f"Erreur lors de la visualisation 3D: {e}")

def visualize_gaussian_fit(surface_data, model):
    """
    Visualise l'ajustement gaussien
    """
    if not surface_data or not model:
        return
    
    try:
        # Données
        radii = surface_data['radii']
        radial_profile = np.nanmean(surface_data['surface'], axis=0)
        radial_profile = np.nan_to_num(radial_profile)
        
        # Paramètres du modèle
        A = model['amplitude']
        sigma = model['sigma']
        z0 = model['baseline']
        r_squared = model['r_squared']
        
        # Courbe ajustée
        y_pred = model['y_pred']
        
        # Visualisation
        plt.figure(figsize=(10, 6))
        plt.plot(radii, radial_profile, 'b-', linewidth=2, label='Profil radial mesuré')
        plt.plot(radii, y_pred, 'r--', linewidth=2, label=f'Ajustement gaussien (R²={r_squared:.4f})')
        
        plt.xlabel('Rayon (pixels)')
        plt.ylabel('Profondeur (pixels)')
        plt.title('Ajustement du Modèle Gaussien')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Afficher les paramètres
        textstr = f'A = {A:.2f}\nσ = {sigma:.2f}\nz₀ = {z0:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=props)
        
        plt.savefig(f"{OUTPUT_DIR}/gaussian_fit.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Ajustement gaussien sauvegardé: {OUTPUT_DIR}/gaussian_fit.png")
    except Exception as e:
        print(f"Erreur lors de la visualisation du modèle: {e}")

def save_results(angles, model, surface_data):
    """
    Sauvegarde les résultats en JSON
    """
    results = {
        'analysis_date': str(datetime.datetime.now()),
        'number_of_images': len(angles),
        'angles_analyzed': angles,
        'model_parameters': model,
        'surface_statistics': {
            'max_depth': float(np.min(surface_data['surface'])) if surface_data else None,
            'surface_area': float(np.sum(np.abs(np.diff(surface_data['surface'], axis=1)))) if surface_data else None
        }
    }
    
    output_file = f"{OUTPUT_DIR}/analysis_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✓ Résultats sauvegardés dans {output_file}")

# Exécution principale
if __name__ == "__main__":
    print("=== ANALYSEUR OCT SIMPLIFIÉ ===")
    print("=" * 40)
    
    # Dossier contenant les images OCT
    oct_folder = "OD/RAD"
    
    # Chercher les images dans ce dossier
    image_paths = glob.glob(f"{oct_folder}/*.tif")
    
    # Si aucune image trouvée, chercher dans d'autres formats
    if not image_paths:
        image_paths = glob.glob(f"{oct_folder}/*.jpg") + glob.glob(f"{oct_folder}/*.png")
    
    # Si aucune image trouvée, chercher dans le dossier courant
    if not image_paths:
        image_paths = glob.glob("*.tif") + glob.glob("*.jpg") + glob.glob("*.png")
    
    # Limiter à 5 images pour les tests
    image_paths = sorted(image_paths)[:5]
    
    if not image_paths:
        print("⚠️ Aucune image OCT trouvée!")
        print("Veuillez placer des images .tif, .jpg ou .png dans le dossier.")
        exit(1)
    
    print(f"Trouvé {len(image_paths)} images OCT:")
    for i, path in enumerate(image_paths):
        print(f"  {i+1}. {Path(path).name}")
    
    # Analyser les images
    success = analyse_oct_images(image_paths)
    
    if success:
        print("\n✅ ANALYSE TERMINÉE AVEC SUCCÈS!")
        print(f"Résultats sauvegardés dans: {OUTPUT_DIR}")
    else:
        print("\n⚠️ ANALYSE INCOMPLÈTE")
        print("Vérifiez que vos images OCT sont correctes.")