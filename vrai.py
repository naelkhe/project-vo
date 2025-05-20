import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from scipy.interpolate import interp1d, RectBivariateSpline
import os
import json
from pathlib import Path

class OCTFoveolarAnalyzer:
    """
    Analyseur pour la modélisation de la dépression fovéolaire en OCT radiale
    """
    
    def __init__(self):
        self.images_data = []
        self.ilm_curves = []
        self.angles = []
        self.center_coordinates = None
        self.surface_3d = None
        self.model_parameters = {}
        
    def load_image(self, image_path, angle=None):
        """
        Charge une image OCT composite (fond d'œil + coupe)
        """
        try:
            # Charger l'image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
            
            # Conversion en niveaux de gris
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extraction de l'angle depuis les métadonnées EXIF ou nom de fichier
            if angle is None:
                angle = self._extract_angle_from_filename(image_path)
            
            self.images_data.append({
                'path': image_path,
                'image': image,
                'gray': gray,
                'angle': angle
            })
            
            print(f"✓ Image chargée: {Path(image_path).name}, angle: {angle}°")
            return True
            
        except Exception as e:
            print(f"✗ Erreur lors du chargement de {image_path}: {e}")
            return False
    
    def _extract_angle_from_filename(self, filename):
        """
        Extrait l'angle depuis le nom du fichier ou métadonnées
        """
        # Logique d'extraction d'angle (à adapter selon votre nomenclature)
        import re
        angle_match = re.search(r'(\d+)', Path(filename).stem)
        if angle_match:
            return int(angle_match.group(1))
        else:
            return len(self.images_data) * 10  # Angle par défaut
    
    def separate_composite_image(self, image):
        """
        Sépare l'image composite en fond d'œil et coupe OCT
        """
        height, width = image.shape[:2]
        
        # Détection automatique de la séparation
        # Recherche de la ligne verticale de séparation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Calcul de variance verticale pour trouver la séparation
        vertical_variance = np.var(gray, axis=0)
        
        # Trouver le minimum local (ligne noire de séparation)
        separation_idx = np.argmin(vertical_variance[width//4:3*width//4]) + width//4
        
        # Séparation des deux parties
        fundus_image = image[:, :separation_idx]
        oct_cross_section = image[:, separation_idx:]
        
        return fundus_image, oct_cross_section
    
    def detect_ilm_boundary(self, oct_image):
        """
        Détecte la limitante interne (ILM) sur la coupe OCT
        """
        # Conversion en niveaux de gris si nécessaire
        if len(oct_image.shape) == 3:
            gray = cv2.cvtColor(oct_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = oct_image.copy()
        
        # Prétraitement pour améliorer le contraste
        enhanced = cv2.equalizeHist(gray)
        
        # Détection de contours avec l'algorithme de Canny
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Filtrage morphologique pour nettoyer
        kernel = np.ones((3,3), np.uint8)
        edges_cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Extraction de la courbe ILM (première interface supérieure)
        ilm_points = []
        
        for x in range(gray.shape[1]):
            # Chercher le premier pixel blanc (contour) depuis le haut
            column = edges_cleaned[:, x]
            nonzero_indices = np.nonzero(column)[0]
            
            if len(nonzero_indices) > 0:
                # Prendre le premier contour significatif
                y_candidates = nonzero_indices[nonzero_indices > gray.shape[0]//4]
                if len(y_candidates) > 0:
                    ilm_points.append((x, y_candidates[0]))
        
        # Lissage de la courbe ILM
        if len(ilm_points) > 10:
            ilm_points = np.array(ilm_points)
            x_coords = ilm_points[:, 0]
            y_coords = ilm_points[:, 1]
            
            # Ajustement polynomial pour lissage
            poly_coeffs = np.polyfit(x_coords, y_coords, 3)
            x_smooth = np.linspace(0, gray.shape[1]-1, gray.shape[1])
            y_smooth = np.polyval(poly_coeffs, x_smooth)
            
            return x_smooth, y_smooth
        else:
            return None, None
    
    def register_images(self):
        """
        Recale les images OCT entre elles
        """
        if len(self.images_data) < 2:
            print("Pas assez d'images pour le recalage")
            return
        
        # Image de référence (première image)
        reference_fundus, _ = self.separate_composite_image(self.images_data[0]['image'])
        ref_gray = cv2.cvtColor(reference_fundus, cv2.COLOR_BGR2GRAY)
        
        # Détecteur de points d'intérêt SIFT
        sift = cv2.SIFT_create()
        ref_kp, ref_desc = sift.detectAndCompute(ref_gray, None)
        
        # Recalage pour chaque image
        for i, data in enumerate(self.images_data[1:], 1):
            fundus, oct_section = self.separate_composite_image(data['image'])
            fund_gray = cv2.cvtColor(fundus, cv2.COLOR_BGR2GRAY)
            
            # Détection des points d'intérêt
            kp, desc = sift.detectAndCompute(fund_gray, None)
            
            # Appariement des descripteurs
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(ref_desc, desc, k=2)
            
            # Filtrage des bons appariements (test de Lowe)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) > 10:
                # Calcul de la transformation géométrique
                src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Estimation de l'homographie
                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                
                # Application de la transformation
                registered_image = cv2.warpPerspective(data['image'], M, 
                                                     (data['image'].shape[1], data['image'].shape[0]))
                
                self.images_data[i]['registered'] = registered_image
                print(f"✓ Image {i} recalée avec {len(good_matches)} points d'appariement")
            else:
                print(f"✗ Pas assez de points d'appariement pour l'image {i}")
    
    def extract_all_ilm_curves(self):
        """
        Extrait les courbes ILM de toutes les images
        """
        self.ilm_curves = []
        self.angles = []
        
        for i, data in enumerate(self.images_data):
            # Utiliser l'image recalée si disponible
            image_to_process = data.get('registered', data['image'])
            
            # Séparer l'image composite
            _, oct_section = self.separate_composite_image(image_to_process)
            
            # Détecter l'ILM
            x_coords, y_coords = self.detect_ilm_boundary(oct_section)
            
            if x_coords is not None:
                self.ilm_curves.append({
                    'x': x_coords,
                    'y': y_coords,
                    'angle': data['angle']
                })
                self.angles.append(data['angle'])
                print(f"✓ Courbe ILM extraite pour l'angle {data['angle']}°")
            else:
                print(f"✗ Échec extraction ILM pour l'angle {data['angle']}°")
    
    def reconstruct_3d_surface(self):
        """
        Reconstruit la surface 3D de la dépression fovéolaire
        """
        if not self.ilm_curves:
            print("Aucune courbe ILM disponible")
            return
        
        # Conversion en coordonnées polaires
        polar_data = []
        
        for curve in self.ilm_curves:
            angle_rad = np.deg2rad(curve['angle'])
            
            # Trouver le centre de la dépression (point le plus bas)
            min_idx = np.argmin(curve['y'])
            center_x = curve['x'][min_idx]
            
            # Conversion en coordonnées polaires relatives au centre
            x_relative = curve['x'] - center_x
            r_coords = np.abs(x_relative)
            z_coords = curve['y'] - curve['y'][min_idx]  # Hauteur relative
            
            polar_data.append({
                'angle': angle_rad,
                'r': r_coords,
                'z': z_coords,
                'center': (center_x, curve['y'][min_idx])
            })
        
        # Interpolation pour créer une surface 3D continue
        # Création d'une grille polaire régulière
        angles_interp = np.linspace(0, 2*np.pi, 360)
        r_max = max([np.max(data['r']) for data in polar_data])
        r_interp = np.linspace(0, r_max, 100)
        
        # Interpolation des données
        surface_3d = np.zeros((len(angles_interp), len(r_interp)))
        
        for i, angle in enumerate(angles_interp):
            # Trouver les angles les plus proches dans les données
            angle_diffs = [abs(angle - data['angle']) for data in polar_data]
            closest_idx = np.argmin(angle_diffs)
            
            # Interpolation radiale pour cet angle
            closest_data = polar_data[closest_idx]
            interp_func = interp1d(closest_data['r'], closest_data['z'], 
                                 kind='cubic', bounds_error=False, fill_value=0)
            surface_3d[i, :] = interp_func(r_interp)
        
        self.surface_3d = {
            'surface': surface_3d,
            'angles': angles_interp,
            'radii': r_interp
        }
        
        print("✓ Surface 3D reconstruite")
    
    def fit_gaussian_model(self):
        """
        Ajuste un modèle gaussien à la surface 3D
        """
        if self.surface_3d is None:
            print("Surface 3D non disponible")
            return
        
        # Paramètres de la surface gaussienne: z = A * exp(-(r²)/(2σ²)) + z₀
        surface = self.surface_3d['surface']
        radii = self.surface_3d['radii']
        
        # Moyenner sur tous les angles pour obtenir un profil radial moyen
        radial_profile = np.mean(surface, axis=0)
        
        # Ajustement gaussien
        from scipy.optimize import curve_fit
        
        def gaussian_surface(r, A, sigma, z0):
            return A * np.exp(-(r**2) / (2 * sigma**2)) + z0
        
        # Estimation initiale des paramètres
        A_init = np.min(radial_profile)  # Profondeur maximale
        sigma_init = radii[len(radii)//4]  # Largeur approximative
        z0_init = np.mean(radial_profile[-10:])  # Niveau de base
        
        try:
            # Ajustement des paramètres
            popt, pcov = curve_fit(gaussian_surface, radii, radial_profile,
                                 p0=[A_init, sigma_init, z0_init])
            
            A_fit, sigma_fit, z0_fit = popt
            
            # Calcul de la qualité de l'ajustement
            y_pred = gaussian_surface(radii, *popt)
            r_squared = 1 - np.sum((radial_profile - y_pred)**2) / np.sum((radial_profile - np.mean(radial_profile))**2)
            
            self.model_parameters = {
                'model_type': 'gaussian',
                'amplitude': A_fit,
                'sigma': sigma_fit,
                'baseline': z0_fit,
                'r_squared': r_squared,
                'rmse': np.sqrt(np.mean((radial_profile - y_pred)**2))
            }
            
            print(f"✓ Modèle gaussien ajusté:")
            print(f"  - Amplitude: {A_fit:.2f}")
            print(f"  - Sigma: {sigma_fit:.2f}")
            print(f"  - Baseline: {z0_fit:.2f}")
            print(f"  - R²: {r_squared:.4f}")
            
        except Exception as e:
            print(f"✗ Erreur lors de l'ajustement gaussien: {e}")
    
    def visualize_results(self, output_dir="results"):
        """
        Génère les visualisations des résultats
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Images avec ILM détectée
        self._plot_ilm_detection(output_dir)
        
        # 2. Surface 3D
        self._plot_3d_surface(output_dir)
        
        # 3. Modèle gaussien
        self._plot_gaussian_fit(output_dir)
        
        # 4. GIF animé des images recalées
        self._create_animated_gif(output_dir)
    
    def _plot_ilm_detection(self, output_dir):
        """
        Visualise la détection ILM sur toutes les images
        """
        fig, axes = plt.subplots(len(self.images_data), 1, figsize=(12, 4*len(self.images_data)))
        if len(self.images_data) == 1:
            axes = [axes]
        
        for i, (data, curve) in enumerate(zip(self.images_data, self.ilm_curves)):
            _, oct_section = self.separate_composite_image(data['image'])
            
            axes[i].imshow(oct_section, cmap='gray')
            axes[i].plot(curve['x'], curve['y'], 'r-', linewidth=2, label='ILM détectée')
            axes[i].set_title(f'Détection ILM - Angle {curve["angle"]}°')
            axes[i].legend()
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ilm_detection.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_3d_surface(self, output_dir):
        """
        Visualise la surface 3D reconstruite
        """
        if self.surface_3d is None:
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Conversion en coordonnées cartésiennes pour l'affichage
        R, THETA = np.meshgrid(self.surface_3d['radii'], self.surface_3d['angles'])
        X = R * np.cos(THETA)
        Y = R * np.sin(THETA)
        Z = self.surface_3d['surface']
        
        # Surface 3D
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_zlabel('Profondeur (pixels)')
        ax.set_title('Surface 3D de la Dépression Fovéolaire')
        
        # Barre de couleurs
        fig.colorbar(surface)
        
        plt.savefig(f"{output_dir}/surface_3d.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gaussian_fit(self, output_dir):
        """
        Visualise l'ajustement gaussien
        """
        if not self.model_parameters:
            return
        
        # Profil radial moyen
        radial_profile = np.mean(self.surface_3d['surface'], axis=0)
        radii = self.surface_3d['radii']
        
        # Modèle gaussien ajusté
        A, sigma, z0 = self.model_parameters['amplitude'], self.model_parameters['sigma'], self.model_parameters['baseline']
        gaussian_fit = A * np.exp(-(radii**2) / (2 * sigma**2)) + z0
        
        plt.figure(figsize=(10, 6))
        plt.plot(radii, radial_profile, 'b-', linewidth=2, label='Profil radial mesuré')
        plt.plot(radii, gaussian_fit, 'r--', linewidth=2, label=f'Ajustement gaussien (R²={self.model_parameters["r_squared"]:.4f})')
        plt.xlabel('Rayon (pixels)')
        plt.ylabel('Profondeur (pixels)')
        plt.title('Ajustement du Modèle Gaussien')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Ajouter les paramètres sur le graphique
        textstr = f'A = {A:.2f}\nσ = {sigma:.2f}\nz₀ = {z0:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.savefig(f"{output_dir}/gaussian_fit.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_animated_gif(self, output_dir):
        """
        Crée un GIF animé des images recalées avec ILM
        """
        images_for_gif = []
        
        for i, (data, curve) in enumerate(zip(self.images_data, self.ilm_curves)):
            # Utiliser l'image recalée si disponible
            image_to_use = data.get('registered', data['image'])
            _, oct_section = self.separate_composite_image(image_to_use)
            
            # Créer une image avec ILM superposée
            oct_with_ilm = oct_section.copy()
            if len(oct_with_ilm.shape) == 3:
                oct_with_ilm = cv2.cvtColor(oct_with_ilm, cv2.COLOR_BGR2RGB)
            else:
                oct_with_ilm = cv2.cvtColor(oct_with_ilm, cv2.COLOR_GRAY2RGB)
            
            # Dessiner la courbe ILM en rouge
            for j in range(len(curve['x'])-1):
                cv2.line(oct_with_ilm, 
                        (int(curve['x'][j]), int(curve['y'][j])),
                        (int(curve['x'][j+1]), int(curve['y'][j+1])),
                        (255, 0, 0), 2)
            
            # Ajouter le texte de l'angle
            cv2.putText(oct_with_ilm, f"Angle: {curve['angle']}°", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            images_for_gif.append(oct_with_ilm)
        
        # Sauvegarder les images individuelles et créer l'info pour le GIF
        for i, img in enumerate(images_for_gif):
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'OCT avec ILM - Angle {self.ilm_curves[i]["angle"]}°')
            plt.savefig(f"{output_dir}/oct_with_ilm_{i:02d}.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Frames du GIF sauvegardées dans {output_dir}")
        print("  Pour créer le GIF, utilisez: imageio ou PIL pour combiner les images oct_with_ilm_*.png")
    
    def save_results(self, output_file="foveolar_analysis_results.json"):
        """
        Sauvegarde tous les résultats en JSON
        """
        results = {
            'analysis_date': str(pd.Timestamp.now()),
            'number_of_images': len(self.images_data),
            'angles_analyzed': self.angles,
            'model_parameters': self.model_parameters,
            'surface_statistics': {
                'max_depth': float(np.min(self.surface_3d['surface'])) if self.surface_3d else None,
                'surface_area': float(np.trapz(np.trapz(self.surface_3d['surface']))) if self.surface_3d else None
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"✓ Résultats sauvegardés dans {output_file}")

# Fonction principale d'utilisation
def analyze_foveolar_depression(image_folder, output_folder="analysis_results"):
    """
    Fonction principale pour analyser la dépression fovéolaire
    
    Args:
        image_folder: Dossier contenant les images OCT
        output_folder: Dossier de sortie pour les résultats
    """
    # Créer l'analyseur
    analyzer = OCTFoveolarAnalyzer()
    
    # Charger toutes les images du dossier
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_folder).glob(f"*{ext}"))
        image_files.extend(Path(image_folder).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"Aucune image trouvée dans {image_folder}")
        return None
    
    # Charger les images
    for image_file in sorted(image_files):
        analyzer.load_image(str(image_file))
    
    # Pipeline d'analyse complet
    print("\n=== PIPELINE D'ANALYSE ===")
    
    # 1. Recalage des images
    print("1. Recalage des images...")
    analyzer.register_images()
    
    # 2. Extraction des courbes ILM
    print("2. Extraction des courbes ILM...")
    analyzer.extract_all_ilm_curves()
    
    # 3. Reconstruction 3D
    print("3. Reconstruction de la surface 3D...")
    analyzer.reconstruct_3d_surface()
    
    # 4. Ajustement du modèle gaussien
    print("4. Ajustement du modèle gaussien...")
    analyzer.fit_gaussian_model()
    
    # 5. Génération des visualisations
    print("5. Génération des visualisations...")
    analyzer.visualize_results(output_folder)
    
    # 6. Sauvegarde des résultats
    print("6. Sauvegarde des résultats...")
    analyzer.save_results(f"{output_folder}/analysis_results.json")
    
    print(f"\n✅ ANALYSE TERMINÉE! Résultats dans: {output_folder}")
    
    return analyzer

# Exemple d'utilisation
if __name__ == "__main__":
    # Analyser un dossier d'images OCT
    # analyzer = analyze_foveolar_depression("path/to/oct/images", "results")
    
    # Ou utilisation étape par étape
    analyzer = OCTFoveolarAnalyzer()
    
    # Charger des images individuelles
    # analyzer.load_image("image1.jpg", angle=0)
    # analyzer.load_image("image2.jpg", angle=30)
    # analyzer.load_image("image3.jpg", angle=60)
    
    print("Pipeline d'analyse OCT prêt à l'utilisation!")