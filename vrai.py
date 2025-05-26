import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import interp1d
import os
import json
from pathlib import Path
import datetime
import glob

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
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
            
            # Conversion en niveaux de gris
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extraction de l'angle depuis les métadonnées EXIF ou nom de fichier
            if angle is None:
                angle = self._extract_angle_from_filename(image_path)
            
            self.images_data.append({
                'path': str(image_path),
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
        # Logique d'extraction d'angle adaptée à vos images OCT
        try:
            import re
            filename_str = str(filename)
            
            # Tenter d'extraire "Q: XX" comme vu dans vos images
            q_match = re.search(r'Q:\s*(\d+)', filename_str)
            if q_match:
                return int(q_match.group(1))
            
            # Tenter d'extraire un numéro de série pour les images RAD
            # Format: CONTROL_CHC_20241022_RAD_OD_XXX.tif
            rad_match = re.search(r'RAD_OD_(\d+)', filename_str)
            if rad_match:
                # Convertir l'index en angle (0-360°)
                index = int(rad_match.group(1))
                total_files = 28  # Nombre total approximatif de fichiers RAD
                angle = int(index * 360 / total_files)
                return angle
            
            # Autres tentatives d'extraction
            angle_match = re.search(r'(\d+)[°\s]', filename_str)
            if angle_match:
                return int(angle_match.group(1))
            
            # Utiliser un angle basé sur l'ordre si pas trouvé
            return len(self.images_data) * 30
        except Exception as e:
            print(f"Erreur lors de l'extraction d'angle: {e}")
            # Sécurité: retourner un angle par défaut
            return len(self.images_data) * 30
    
    def separate_composite_image(self, image):
        """
        Sépare l'image composite en fond d'œil et coupe OCT
        """
        height, width = image.shape[:2]
        
        # Détection automatique de la séparation
        # Recherche de la ligne verticale de séparation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Vérifier si l'image est probablement composite
        if width > height * 1.5:  # Image large, probablement composite
            # Calcul de variance verticale pour trouver la séparation
            vertical_variance = np.var(gray, axis=0)
            
            # Trouver le minimum local (ligne noire de séparation)
            # Limiter la recherche au milieu de l'image
            search_start = width // 4
            search_end = 3 * width // 4
            
            if search_end <= search_start:
                separation_idx = width // 2  # Fallback
            else:
                middle_section = vertical_variance[search_start:search_end]
                separation_idx = np.argmin(middle_section) + search_start
            
            # Séparation des deux parties
            fundus_image = image[:, :separation_idx]
            oct_cross_section = image[:, separation_idx:]
        else:
            # Image non composite, considérer l'image entière comme OCT
            fundus_image = np.zeros((height, 1, 3), dtype=np.uint8)
            oct_cross_section = image
        
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
        
        try:
            # Amélioration du contraste pour mieux voir les détails
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Filtrage bilatéral pour réduire le bruit tout en préservant les bords
            filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Calculer le gradient vertical pour détecter les transitions (noir->blanc)
            grad_y = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
            
            # Extraire la courbe ILM
            height, width = gray.shape
            ilm_points = []
            
            # Définir une région d'intérêt excluant les marges supérieures
            roi_top = height // 5       # Ignorer le premier cinquième (noir)
            roi_bottom = height // 2    # Limiter à la moitié supérieure
            
            for x in range(width):
                # Examiner seulement la région d'intérêt
                column = grad_y[roi_top:roi_bottom, x]
                
                # Chercher le point avec le plus fort gradient positif (transition noir->blanc)
                max_val_indices = np.argmax(column)
                
                if max_val_indices.size > 0:
                    y = roi_top + max_val_indices
                    # Vérifier que le gradient est assez fort pour être significatif
                    if column[max_val_indices] > np.std(column) * 2:
                        ilm_points.append((x, y))
            
            # Vérifier si assez de points ont été trouvés
            if len(ilm_points) > width // 4:  # Au moins 25% de la largeur
                # Conversion en array numpy
                ilm_points = np.array(ilm_points)
                
                # Tri par position X
                sort_idx = np.argsort(ilm_points[:, 0])
                x_coords = ilm_points[sort_idx, 0]
                y_coords = ilm_points[sort_idx, 1]
                
                # Filtrage des outliers avec un filtre médian
                y_filtered = ndimage.median_filter(y_coords, size=5)
                
                # Interpolation polynomiale pour lisser la courbe
                poly_coeffs = np.polyfit(x_coords, y_filtered, 4)  # 4ème degré pour plus de flexibilité
                x_smooth = np.linspace(0, width-1, width)
                y_smooth = np.polyval(poly_coeffs, x_smooth)
                
                return x_smooth, y_smooth
            else:
                print(f"Pas assez de points ILM trouvés ({len(ilm_points)})")
                return None, None
                
        except Exception as e:
            print(f"Erreur dans la détection ILM: {e}")
            return None, None
    
    def register_images(self):
        """
        Recale les images OCT entre elles
        """
        if len(self.images_data) < 2:
            print("Pas assez d'images pour le recalage")
            return
        
        try:
            # Image de référence (première image)
            reference_fundus, _ = self.separate_composite_image(self.images_data[0]['image'])
            ref_gray = cv2.cvtColor(reference_fundus, cv2.COLOR_BGR2GRAY)
            
            # Tentative d'utiliser SIFT pour le recalage
            try:
                # Détecteur de points d'intérêt SIFT
                sift = cv2.SIFT_create()
                ref_kp, ref_desc = sift.detectAndCompute(ref_gray, None)
                
                # Recalage pour chaque image
                for i, data in enumerate(self.images_data[1:], 1):
                    fundus, oct_section = self.separate_composite_image(data['image'])
                    fund_gray = cv2.cvtColor(fundus, cv2.COLOR_BGR2GRAY)
                    
                    # Détection des points d'intérêt
                    kp, desc = sift.detectAndCompute(fund_gray, None)
                    
                    if desc is None or len(desc) < 2 or ref_desc is None or len(ref_desc) < 2:
                        print(f"Pas assez de descripteurs pour l'image {i}")
                        self.images_data[i]['registered'] = data['image']
                        continue
                    
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
                        # Utiliser l'image originale comme fallback
                        self.images_data[i]['registered'] = data['image']
                
            except Exception as e:
                print(f"Erreur lors du recalage SIFT: {e}")
                # Méthode alternative: recalage simple basé sur ECC
                self._register_images_fallback()
        
        except Exception as e:
            print(f"Erreur lors du recalage: {e}. Continuons sans recalage.")
            # Ne pas bloquer l'analyse si le recalage échoue
            for i, data in enumerate(self.images_data[1:], 1):
                self.images_data[i]['registered'] = data['image']
    
    def _register_images_fallback(self):
        """
        Méthode de recalage alternative basée sur ECC (Enhanced Correlation Coefficient)
        """
        print("Tentative de recalage alternatif...")
        
        # Image de référence
        reference_fundus, _ = self.separate_composite_image(self.images_data[0]['image'])
        ref_gray = cv2.cvtColor(reference_fundus, cv2.COLOR_BGR2GRAY) if len(reference_fundus.shape) == 3 else reference_fundus
        
        # Paramètres ECC
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        warp_mode = cv2.MOTION_TRANSLATION  # Translation simple
        
        for i, data in enumerate(self.images_data[1:], 1):
            fundus, _ = self.separate_composite_image(data['image'])
            fund_gray = cv2.cvtColor(fundus, cv2.COLOR_BGR2GRAY) if len(fundus.shape) == 3 else fundus
            
            try:
                # Matrice de transformation (pour translation)
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                
                # Recalage ECC
                _, warp_matrix = cv2.findTransformECC(ref_gray, fund_gray, warp_matrix, warp_mode, criteria)
                
                # Application de la transformation
                height, width = data['image'].shape[:2]
                registered_image = cv2.warpAffine(data['image'], warp_matrix, (width, height), 
                                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                
                self.images_data[i]['registered'] = registered_image
                print(f"✓ Image {i} recalée avec méthode ECC")
                
            except Exception as e:
                print(f"✗ Échec du recalage alternatif pour l'image {i}: {e}")
                self.images_data[i]['registered'] = data['image']
    
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
            print(f"Détection ILM pour l'image {i+1}...")
            x_coords, y_coords = self.detect_ilm_boundary(oct_section)
            
            if x_coords is not None and y_coords is not None:
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
            
            # CORRECTION: Assurer des valeurs r uniques en éliminant les doublons
            # Trouver les indices uniques (garder première occurrence)
            _, unique_indices = np.unique(r_coords, return_index=True)
            unique_indices = np.sort(unique_indices)  # Trier pour maintenir l'ordre
            
            # Ne garder que les valeurs uniques
            r_unique = r_coords[unique_indices]
            z_unique = z_coords[unique_indices]
            
            polar_data.append({
                'angle': angle_rad,
                'r': r_unique,
                'z': z_unique,
                'center': (center_x, curve['y'][min_idx])
            })
        
        # Vérification qu'il y a suffisamment de données
        if len(polar_data) < 2:
            print("Pas assez de coupes pour une reconstruction 3D fiable")
            # Si une seule courbe, créer une surface de révolution
            if len(polar_data) == 1:
                print("Création d'une surface de révolution à partir de la coupe unique")
                single_data = polar_data[0]
                
                # Créer des angles intermédiaires
                angles_interp = np.linspace(0, 2*np.pi, 72)
                r_interp = single_data['r']
                
                # Surface de révolution
                surface_3d = np.zeros((len(angles_interp), len(r_interp)))
                for i in range(len(angles_interp)):
                    surface_3d[i, :] = single_data['z']
                
                self.surface_3d = {
                    'surface': surface_3d,
                    'angles': angles_interp,
                    'radii': r_interp
                }
                print("✓ Surface de révolution créée (approximation)")
            return
        
        # Interpolation pour créer une surface 3D continue
        # Création d'une grille polaire régulière
        angles_interp = np.linspace(0, 2*np.pi, 72)  # 72 angles = 5 degrés par pas
        r_max = max([np.max(data['r']) for data in polar_data])
        r_interp = np.linspace(0, r_max, 100)
        
        # Interpolation des données
        surface_3d = np.zeros((len(angles_interp), len(r_interp)))
        
        # Trouver les angles existants
        existing_angles = np.array([data['angle'] for data in polar_data])
        
        for i, angle in enumerate(angles_interp):
            # Trouver les angles les plus proches dans les données
            angle_diffs = np.abs(np.mod(angle - existing_angles + np.pi, 2*np.pi) - np.pi)
            closest_idx = np.argmin(angle_diffs)
            
            # Interpolation radiale pour cet angle
            closest_data = polar_data[closest_idx]
            
            # Vérification des données avant interpolation
            if len(closest_data['r']) < 4:  # Besoin d'au moins 4 points pour une spline cubique
                # Utiliser une interpolation linéaire si peu de points
                kind = 'linear'
            else:
                kind = 'linear'  # Changé de 'cubic' à 'linear' pour éviter les erreurs
            
            # Ensure data is valid for interpolation
            valid_indices = np.isfinite(closest_data['z'])
            if np.sum(valid_indices) > 3:  # Need at least a few points
                try:
                    interp_func = interp1d(
                        closest_data['r'][valid_indices], 
                        closest_data['z'][valid_indices], 
                        kind=kind,
                        bounds_error=False, 
                        fill_value=0
                    )
                    surface_3d[i, :] = interp_func(r_interp)
                except Exception as e:
                    print(f"Erreur d'interpolation: {e}")
                    surface_3d[i, :] = 0
            else:
                surface_3d[i, :] = 0
        
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
        radial_profile = np.nanmean(surface, axis=0)
        
        # Remplacer NaN par 0
        radial_profile = np.nan_to_num(radial_profile)
        
        # Ajustement gaussien
        from scipy.optimize import curve_fit
        
        def gaussian_surface(r, A, sigma, z0):
            return A * np.exp(-(r**2) / (2 * sigma**2)) + z0
        
        # Estimation initiale des paramètres
        A_init = np.min(radial_profile) - np.max(radial_profile)  # Profondeur (valeur négative)
        sigma_init = radii[len(radii)//4]  # Largeur approximative
        z0_init = np.max(radial_profile)   # Niveau de base (hauteur maximale)
        
        try:
            # Ajustement des paramètres
            popt, pcov = curve_fit(
                gaussian_surface, 
                radii, 
                radial_profile,
                p0=[A_init, sigma_init, z0_init],
                bounds=([A_init*1.5, 0, z0_init*0.5], [A_init*0.5, radii[-1], z0_init*1.5])
            )
            
            A_fit, sigma_fit, z0_fit = popt
            
            # Calcul de la qualité de l'ajustement
            y_pred = gaussian_surface(radii, *popt)
            ss_total = np.sum((radial_profile - np.mean(radial_profile))**2)
            ss_residual = np.sum((radial_profile - y_pred)**2)
            
            r_squared = 1 - ss_residual / ss_total if ss_total > 0 else 0
            
            self.model_parameters = {
                'model_type': 'gaussian',
                'amplitude': float(A_fit),
                'sigma': float(sigma_fit),
                'baseline': float(z0_fit),
                'r_squared': float(r_squared),
                'rmse': float(np.sqrt(np.mean((radial_profile - y_pred)**2)))
            }
            
            print(f"✓ Modèle gaussien ajusté:")
            print(f"  - Amplitude: {A_fit:.2f}")
            print(f"  - Sigma: {sigma_fit:.2f}")
            print(f"  - Baseline: {z0_fit:.2f}")
            print(f"  - R²: {r_squared:.4f}")
            
        except Exception as e:
            print(f"✗ Erreur lors de l'ajustement gaussien: {e}")
            # Paramètres par défaut en cas d'échec
            self.model_parameters = {
                'model_type': 'gaussian',
                'amplitude': float(A_init),
                'sigma': float(sigma_init),
                'baseline': float(z0_init),
                'r_squared': 0.0,
                'rmse': 0.0,
                'error': str(e)
            }
    
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
        if not self.ilm_curves or len(self.ilm_curves) == 0:
            print("Données ILM incomplètes, impossible de générer la visualisation")
            return
        
        try:
            fig, axes = plt.subplots(len(self.ilm_curves), 1, figsize=(12, 4*len(self.ilm_curves)))
            if len(self.ilm_curves) == 1:
                axes = [axes]
            
            for i, curve in enumerate(self.ilm_curves):
                if i >= len(self.images_data):
                    continue
                    
                data = self.images_data[i]
                _, oct_section = self.separate_composite_image(data['image'])
                
                # Conversion en RGB pour l'affichage
                if len(oct_section.shape) == 3:
                    oct_display = cv2.cvtColor(oct_section, cv2.COLOR_BGR2RGB)
                else:
                    oct_display = cv2.cvtColor(oct_section, cv2.COLOR_GRAY2RGB)
                
                axes[i].imshow(oct_display)
                axes[i].plot(curve['x'], curve['y'], 'r-', linewidth=2, label='ILM détectée')
                axes[i].set_title(f'Détection ILM - Angle {curve["angle"]}°')
                axes[i].legend()
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/ilm_detection.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Visualisation ILM sauvegardée: {output_dir}/ilm_detection.png")
        except Exception as e:
            print(f"Erreur lors de la création de la visualisation ILM: {e}")
    
    def _plot_3d_surface(self, output_dir):
        """
        Visualise la surface 3D reconstruite
        """
        if self.surface_3d is None:
            print("Surface 3D non disponible, impossible de générer la visualisation")
            return
        
        try:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Conversion en coordonnées cartésiennes pour l'affichage
            R, THETA = np.meshgrid(self.surface_3d['radii'], self.surface_3d['angles'])
            X = R * np.cos(THETA)
            Y = R * np.sin(THETA)
            Z = self.surface_3d['surface']
            
            # Surface 3D
            surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_zlabel('Profondeur (pixels)')
            ax.set_title('Surface 3D de la Dépression Fovéolaire')
            
            # Barre de couleurs
            fig.colorbar(surface)
            
            plt.savefig(f"{output_dir}/surface_3d.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Visualisation 3D sauvegardée: {output_dir}/surface_3d.png")
            
        except Exception as e:
            print(f"✗ Erreur lors de la création de la visualisation 3D: {e}")
    
    def _plot_gaussian_fit(self, output_dir):
        """
        Visualise l'ajustement gaussien
        """
        if not self.model_parameters or self.surface_3d is None:
            print("Modèle ou surface non disponibles, impossible de générer la visualisation")
            return
        
        try:
            # Profil radial moyen
            radial_profile = np.nanmean(self.surface_3d['surface'], axis=0)
            radial_profile = np.nan_to_num(radial_profile)
            radii = self.surface_3d['radii']
            
            # Modèle gaussien ajusté
            A = self.model_parameters.get('amplitude', 0)
            sigma = self.model_parameters.get('sigma', 1)
            z0 = self.model_parameters.get('baseline', 0)
            r_squared = self.model_parameters.get('r_squared', 0)
            
            gaussian_fit = A * np.exp(-(radii**2) / (2 * sigma**2)) + z0
            
            plt.figure(figsize=(10, 6))
            plt.plot(radii, radial_profile, 'b-', linewidth=2, label='Profil radial mesuré')
            plt.plot(radii, gaussian_fit, 'r--', linewidth=2, label=f'Ajustement gaussien (R²={r_squared:.4f})')
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
            
            print(f"✓ Visualisation ajustement gaussien sauvegardée: {output_dir}/gaussian_fit.png")
            
        except Exception as e:
            print(f"✗ Erreur lors de la création de la visualisation du modèle: {e}")
    
    def _create_animated_gif(self, output_dir):
        """
        Crée des images individuelles pour animation
        """
        if not self.ilm_curves:
            print("Données ILM non disponibles, impossible de générer les images pour GIF")
            return
        
        try:
            # Créer un sous-dossier pour les frames
            frames_dir = f"{output_dir}/frames"
            os.makedirs(frames_dir, exist_ok=True)
            
            for i, curve in enumerate(self.ilm_curves):
                if i >= len(self.images_data):
                    continue
                
                data = self.images_data[i]
                # Utiliser l'image recalée si disponible
                image_to_use = data.get('registered', data['image'])
                _, oct_section = self.separate_composite_image(image_to_use)
                
                # Créer une image avec ILM superposée
                if len(oct_section.shape) == 3:
                    oct_with_ilm = cv2.cvtColor(oct_section, cv2.COLOR_BGR2RGB)
                else:
                    oct_with_ilm = cv2.cvtColor(oct_section, cv2.COLOR_GRAY2RGB)
                
                # Dessiner la courbe ILM en rouge
                x_coords = curve['x'].astype(int)
                y_coords = curve['y'].astype(int)
                for j in range(len(x_coords)-1):
                    x1, y1 = x_coords[j], y_coords[j]
                    x2, y2 = x_coords[j+1], y_coords[j+1]
                    # Vérifier que les points sont dans les limites
                    if (0 <= x1 < oct_with_ilm.shape[1] and 0 <= y1 < oct_with_ilm.shape[0] and
                        0 <= x2 < oct_with_ilm.shape[1] and 0 <= y2 < oct_with_ilm.shape[0]):
                        cv2.line(oct_with_ilm, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Ajouter le texte de l'angle
                cv2.putText(oct_with_ilm, f"Angle: {curve['angle']}°", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Sauvegarder l'image
                plt.figure(figsize=(10, 6))
                plt.imshow(oct_with_ilm)
                plt.axis('off')
                plt.title(f'OCT avec ILM - Angle {curve["angle"]}°')
                plt.savefig(f"{frames_dir}/oct_with_ilm_{i:02d}.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"✓ Frames pour GIF sauvegardées dans {frames_dir}")
            
            # Tenter de créer le GIF si le module est disponible
            try:
                from PIL import Image
                
                frames = []
                frame_files = sorted(Path(frames_dir).glob("oct_with_ilm_*.png"))
                
                for frame_file in frame_files:
                    img = Image.open(frame_file)
                    frames.append(img.copy())
                    img.close()
                
                if frames:
                    frames[0].save(
                        f"{output_dir}/oct_animation.gif",
                        format='GIF',
                        append_images=frames[1:],
                        save_all=True,
                        duration=500,  # ms par frame
                        loop=0  # boucle infinie
                    )
                    print(f"✓ GIF animé créé: {output_dir}/oct_animation.gif")
            except Exception as e:
                print(f"Note: Impossible de créer le GIF animé automatiquement: {e}")
                print("  Vous pouvez utiliser un outil externe pour combiner les images.")
                
        except Exception as e:
            print(f"✗ Erreur lors de la création des images pour GIF: {e}")
    
    def save_results(self, output_file="foveolar_analysis_results.json"):
        """
        Sauvegarde tous les résultats en JSON
        """
        results = {
            'analysis_date': str(datetime.datetime.now()),
            'number_of_images': len(self.images_data),
            'angles_analyzed': self.angles,
            'model_parameters': self.model_parameters,
            'surface_statistics': {
                'max_depth': float(np.min(self.surface_3d['surface'])) if self.surface_3d else None,
                'surface_area': float(np.sum(np.abs(np.diff(self.surface_3d['surface'], axis=1)))) if self.surface_3d else None
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"✓ Résultats sauvegardés dans {output_file}")


# Programme principal d'exécution
if __name__ == "__main__":
    print("=== ANALYSEUR DE DÉPRESSION FOVÉOLAIRE OCT ===")
    print("=" * 50)
    
    # Créer un analyseur
    analyzer = OCTFoveolarAnalyzer()
    
    # Spécifier le chemin du dossier RAD
    rad_folder = "OD/RAD"  # Chemin relatif au dossier du script
    
    # Utiliser Path pour trouver toutes les images TIFF dans le dossier RAD
    image_paths = glob.glob(f"{rad_folder}/CONTROL_CHC_20241022_RAD_OD_*.tif")
    
    # Limiter à 5 images pour le test initial
    image_paths = sorted(image_paths)[:5]  # Prendre les 5 premières images
    
    print(f"Images trouvées: {len(image_paths)}")
    
    # Charger les images
    for image_path in image_paths:
        print(f"Chargement de: {image_path}")
        analyzer.load_image(image_path)
    
    if len(analyzer.images_data) == 0:
        print("Aucune image n'a pu être chargée!")
        exit(1)
    
    # Créer le dossier de résultats
    output_dir = "resultats_oct"
    
    # Exécuter le pipeline d'analyse
    print("\n=== PIPELINE D'ANALYSE ===")
    
    # 1. Recalage des images
    print("\n1. Recalage des images...")
    analyzer.register_images()
    
    # 2. Extraction des courbes ILM
    print("\n2. Extraction des courbes ILM...")
    analyzer.extract_all_ilm_curves()
    
    # 3. Reconstruction 3D  
    print("\n3. Reconstruction de la surface 3D...")
    analyzer.reconstruct_3d_surface()
    
    # 4. Ajustement du modèle gaussien
    print("\n4. Ajustement du modèle gaussien...")
    analyzer.fit_gaussian_model()
    
    # 5. Visualisation des résultats
    print("\n5. Génération des visualisations...")
    analyzer.visualize_results(output_dir)
    
    # 6. Sauvegarde des résultats
    print("\n6. Sauvegarde des résultats...")
    analyzer.save_results(f"{output_dir}/analysis_results.json")
    
    print(f"\n✅ ANALYSE TERMINÉE!")
    print(f"Les résultats sont disponibles dans le dossier: {output_dir}")