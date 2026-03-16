import cv2
import mediapipe as mp
import time

# Constant for how many faces to track
NUM_FACE = 2

class FaceLandMarks():
    def __init__(self, staticMode=False, maxFace=NUM_FACE, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFace = maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        
        # Initialize FaceMesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode, 
            max_num_faces=self.maxFace, 
            min_detection_confidence=self.minDetectionCon, 
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceLandmark(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        faces = []

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    # Pass 'None' for the connections parameter to remove lines
                    self.mpDraw.draw_landmarks(
                        img, 
                        faceLms, 
                        None,  # This removes the lines/connections
                        self.drawSpec, 
                        self.drawSpec
                    )

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    pTime = 0
    detector = FaceLandMarks()
    
    print("Press 'q' to quit.")
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img, faces = detector.findFaceLandmark(img)
        
        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Mesh Stream", img)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()