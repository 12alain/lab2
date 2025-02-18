import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_match_features(img1, img2):
    """Détecte et associe les points d'intérêt entre deux images en utilisant ORB."""
    orb = cv2.ORB_create(50000)  # Augmenter le nombre de points détectés
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        raise ValueError("Aucune caractéristique détectée dans l'une des images.")

    # Utilisation d'un matcher basé sur Hamming distance (adapté à ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    return kp1, kp2, matches

def find_homography(kp1, kp2, matches, reproj_thresh=4.0):
    """Calcule l'homographie basée sur les correspondances de points clés."""
    if len(matches) < 4:
        raise ValueError("Pas assez de correspondances pour calculer l'homographie.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, reproj_thresh)
    return H, status

def stitch_images(img1, img2, H, blend_mode='average'):
    """Assemble deux images en utilisant une transformation homographique."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Transformer img2 dans le repère d'img1
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_img2, H)

    all_corners = np.vstack((transformed_corners.reshape(-1, 2), np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]])))
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    # Translation pour éviter des pixels négatifs
    translation_matrix = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    H_translated = translation_matrix @ H

    # Warp de img2 pour s'aligner avec img1
    result = cv2.warpPerspective(img2, H_translated, (xmax - xmin, ymax - ymin))

    # Placement de img1 dans le canevas résultant
    img1_transformed = np.zeros_like(result)
    ymin_offset = max(0, -ymin)
    xmin_offset = max(0, -xmin)
    img1_transformed[ymin_offset:ymin_offset+h1, xmin_offset:xmin_offset+w1] = img1

    # Fusion des images
    if blend_mode == 'average':
        mask1 = (result != 0).astype(np.float32)
        mask2 = (img1_transformed != 0).astype(np.float32)
        blended = (result.astype(np.float32) + img1_transformed.astype(np.float32)) / (mask1 + mask2 + 1e-5)
        return blended.astype(np.uint8)

    return result

def mosaic(img1_path, img2_path, blend_mode='average'):
    """Assemble deux images en une mosaïque."""
    
    # Vérifier si img1 est déjà une image numpy (cas d'une fusion précédente)
    if isinstance(img1_path, str):
        img1 = cv2.imread(img1_path)
    else:
        img1 = img1_path  # Si déjà une image en mémoire

    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Impossible de charger les images : {img1_path}, {img2_path}")

    kp1, kp2, matches = detect_and_match_features(img1, img2)
    H, _ = find_homography(kp1, kp2, matches)

    return stitch_images(img1, img2, H, blend_mode)

if __name__ == "__main__":
    # Liste des images à fusionner (3 images dans cet exemple)
    image_paths = [
       
        "/kaggle/input/dataorignale/Images/2/batiment A/1.JPG",
        "/kaggle/input/dataorignale/Images/2/batiment A/2.JPG",
        "/kaggle/input/dataorignale/Images/2/batiment A/3.JPG"
        
    ]

    # Initialiser la mosaïque avec la première image
    result = cv2.imread(image_paths[0])

    # Fusionner les images une par une
    for img_path in image_paths[1:]:
        result = mosaic(result, img_path, blend_mode="average")

    # Sauvegarde du résultat final
    cv2.imwrite("mosaic_result.jpg", result)

    # Affichage du résultat final
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
