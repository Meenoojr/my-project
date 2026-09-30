import re
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from django.core.files.uploadedfile import UploadedFile
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

from .models import CarRegisteration


# =============================================================================
# TESSERACT CONFIGURATION
# =============================================================================

if sys.platform == "win32":
    try:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
    except ImportError:
        pass


# =============================================================================
# NIGERIAN STATES
# =============================================================================

NIGERIAN_STATES = {
    "LAGOS": ("Lagos State", "Centre of Excellence"),
    "ABUJA": ("FCT Abuja", "Centre of Unity"),
    "FCT": ("FCT Abuja", "Centre of Unity"),
    "KANO": ("Kano State", "Centre of Commerce"),
    "KADUNA": ("Kaduna State", "State of Dialogue"),
    "RIVERS": ("Rivers State", "Treasure Base of the Nation"),
    "OGUN": ("Ogun State", "Gateway State"),
    "OYO": ("Oyo State", "Pace Setter State"),
    "ANAMBRA": ("Anambra State", "Light of the Nation"),
    "ENUGU": ("Enugu State", "Coal City State"),
    "DELTA": ("Delta State", "The Big Heart"),
    "ONDO": ("Ondo State", "Sunshine State"),
    "BENUE": ("Benue State", "Food Basket of the Nation"),
    "KOGI": ("Kogi State", "Confluence State"),
    "PLATEAU": ("Plateau State", "Home of Peace and Tourism"),
    "BAUCHI": ("Bauchi State", "Pearl of Tourism"),
    "GOMBE": ("Gombe State", "Jewel in the Savanna"),
    "YOBE": ("Yobe State", "Pride of the Sahara"),
    "BORNO": ("Borno State", "Home of Peace"),
    "ADAMAWA": ("Adamawa State", "Land of Beauty"),
    "TARABA": ("Taraba State", "Nature's Gift to the Nation"),
    "NASARAWA": ("Nasarawa State", "Home of Solid Minerals"),
    "NIGER": ("Niger State", "Power State"),
    "KWARA": ("Kwara State", "State of Harmony"),
    "EKITI": ("Ekiti State", "Land of Honour and Integrity"),
    "OSUN": ("Osun State", "State of the Living Spring"),
    "EDO": ("Edo State", "Heartbeat of the Nation"),
    "IMO": ("Imo State", "Eastern Heartland"),
    "ABIA": ("Abia State", "God's Own State"),
    "EBONYI": ("Ebonyi State", "Salt of the Nation"),
    "BAYELSA": ("Bayelsa State", "Glory of All Lands"),
    "CROSS RIVER": (
        "Cross River State",
        "The Peoples Paradise",
    ),
    "AKWA IBOM": (
        "Akwa Ibom State",
        "Land of Promise",
    ),
    "KEBBI": ("Kebbi State", "Land of Equity"),
    "SOKOTO": ("Sokoto State", "Seat of the Caliphate"),
    "ZAMFARA": ("Zamfara State", "Farming is Our Pride"),
    "KATSINA": ("Katsina State", "Home of Hospitality"),
    "JIGAWA": ("Jigawa State", "Land of Opportunities"),
}


# =============================================================================
# NIGERIAN PLATE PREFIX → STATE
# =============================================================================

PREFIX_TO_STATE = {

    # Lagos
    "AA": "LAGOS",
    "AB": "LAGOS",
    "AC": "LAGOS",
    "AD": "LAGOS",
    "AE": "LAGOS",
    "AF": "LAGOS",
    "AG": "LAGOS",
    "AH": "LAGOS",

    "GGE": "LAGOS",
    "KJA": "LAGOS",
    "LND": "LAGOS",
    "LSD": "LAGOS",
    "APP": "LAGOS",
    "EPE": "LAGOS",
    "FKJ": "LAGOS",

    # FCT Abuja
    "ABJ": "ABUJA",
    "FCT": "ABUJA",

    # Kano
    "KN": "KANO",
    "KNA": "KANO",

    # Kaduna
    "KD": "KADUNA",
    "KDA": "KADUNA",

    # Rivers
    "RI": "RIVERS",
    "RSH": "RIVERS",

    # Ogun
    "OG": "OGUN",
    "OGD": "OGUN",

    # Oyo
    "OY": "OYO",

    # Anambra
    "AN": "ANAMBRA",

    # Enugu
    "EN": "ENUGU",

    # Delta
    "DL": "DELTA",

    # Ondo
    "ON": "ONDO",

    # Kogi
    "KG": "KOGI",

    # Niger
    "MN": "NIGER",

    # Kwara
    "KW": "KWARA",

    # Edo
    "BE": "EDO",

    # Imo
    "IM": "IMO",

    # Abia
    "ABI": "ABIA",

    # Plateau
    "PL": "PLATEAU",
}


# =============================================================================
# OCR CHARACTER CORRECTIONS
# =============================================================================

# Characters that OCR frequently reads instead of letters
LETTER_FIXES = {
    "0": "O",
    "1": "I",
    "5": "S",
    "6": "G",
    "8": "B",
    "2": "Z",
    "4": "A",
}


# Characters that OCR frequently reads instead of numbers
DIGIT_FIXES = {
    "O": "0",
    "I": "1",
    "S": "5",
    "G": "6",
    "B": "8",
    "Z": "2",
    "A": "4",
    "D": "0",
    "Q": "0",
    "E": "3",
}


# =============================================================================
# ALPR SERVICE
# =============================================================================

class ALPRService:
    """
    Nigerian Automatic License Plate Recognition service.

    Expected Nigerian plate structure:

        ABC-123-AA

    Example:

        FKJ-254-XA

    Pipeline:

        Image
          ↓
        YOLO plate detection
          ↓
        Plate crop
          ↓
        Multiple OCR preprocessing variants
          ↓
        EasyOCR + Tesseract
          ↓
        Nigerian plate structure correction
          ↓
        Prefix/state validation
          ↓
        Database lookup
    """

    MODEL_NAMES = [
        "YOLOv8",
        "YOLOv9",
        "YOLOv10",
        "Faster R-CNN",
        "SSD",
    ]

    _detector = None
    _detector_model_name = None
    _ocr_reader = None

    # -------------------------------------------------------------------------
    # Nigerian plate pattern
    # -------------------------------------------------------------------------

    # Main Nigerian format:
    #
    #   ABC123DE
    #
    # 3 letters + 3 digits + 2 letters

    _PLATE_RE = re.compile(
        r"\b([A-Z]{3}[\s\-]?\d{3}[\s\-]?[A-Z]{2})\b"
    )

    _LOOSE_PLATE_RE = re.compile(
        r"\b([A-Z0-9]{5,10})\b"
    )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    @classmethod
    def process_upload(
        cls,
        uploaded_file: UploadedFile,
        model_name: str,
    ) -> Dict[str, object]:

        if not uploaded_file:
            raise ValueError("No uploaded file provided")

        if model_name not in cls.MODEL_NAMES:
            raise ValueError(
                f"Unsupported model: {model_name}"
            )

        uploaded_file.seek(0)

        image = Image.open(uploaded_file).convert("RGB")

        width, height = image.size

        raw_text, plate_number, confidence, status = (
            cls._run_pipeline_full(
                image,
                model_name,
                width,
                height,
            )
        )

        if status == "fallback" or not plate_number:
            return {
                "model_name": model_name,
                "plate_number": None,
                "confidence": 0.0,
                "status": "no-plate-detected",
                "country": None,
                "state": None,
                "slogan": None,
                "raw_text": None,
                "record_found": False,
                "vehicle_owner": None,
                "vehicle_type": None,
                "vehicle_model": None,
                "vehicle_color": None,
                "registration_number": None,
                "image_width": width,
                "image_height": height,
                "ocr_unavailable": not cls._ocr_available(),
            }

        plate_info = cls._extract_plate_info(
            raw_text,
            plate_number,
        )

        record = cls._lookup_registered_vehicle(
            plate_number
        )

        result = {
            "model_name": model_name,
            "plate_number": plate_number,
            "confidence": round(confidence, 3),
            "status": status,
            "country": plate_info.get("country"),
            "state": plate_info.get("state"),
            "slogan": plate_info.get("slogan"),
            "raw_text": plate_info.get("raw_text"),
            "record_found": False,
            "vehicle_owner": None,
            "vehicle_type": None,
            "vehicle_model": None,
            "vehicle_color": None,
            "registration_number": None,
            "image_width": width,
            "image_height": height,
            "ocr_unavailable": False,
        }

        if record is not None:

            result.update({
                "record_found": True,
                "status": "registered",
                "vehicle_owner": record["owner"],
                "vehicle_type": record["vehicle_type"],
                "vehicle_model": record["vehicle_model"],
                "vehicle_color": record["vehicle_color"],
                "registration_number": record["registration_number"],
            })

        else:
            result["status"] = "not-registered"

        return result

    # =========================================================================
    # MODEL LIST
    # =========================================================================

    @classmethod
    def get_available_models(cls) -> List[str]:
        return list(cls.MODEL_NAMES)

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    @classmethod
    def _run_pipeline_full(
        cls,
        image: Any,
        model_name: str,
        width: int,
        height: int,
    ) -> Tuple[str, str, float, str]:

        # ---------------------------------------------------------------------
        # 1. Try YOLO plate detection
        # ---------------------------------------------------------------------

        detection_result = cls._detect_plate_region(
            image,
            model_name,
        )

        if detection_result:

            raw_text, plate_text, confidence = (
                detection_result
            )

            if plate_text:

                return (
                    raw_text,
                    plate_text,
                    confidence,
                    "ocr-detected",
                )

        # ---------------------------------------------------------------------
        # 2. OCR the full image
        # ---------------------------------------------------------------------

        raw_text, plate_text, confidence = (
            cls._ocr_full_image_with_raw(image)
        )

        if plate_text:

            score = cls._score_plate(
                plate_text
            )

            if score >= 5:

                return (
                    raw_text,
                    plate_text,
                    confidence,
                    "ocr-direct",
                )

        # ---------------------------------------------------------------------
        # 3. Last OCR attempt using plate-focused preprocessing
        # ---------------------------------------------------------------------

        plate_text, confidence, raw_text = (
            cls._plate_focused_ocr(image)
        )

        if plate_text:

            return (
                raw_text,
                plate_text,
                confidence,
                "ocr-plate-focused",
            )

        # ---------------------------------------------------------------------
        # 4. Fallback
        # ---------------------------------------------------------------------

        plate, conf, status = (
            cls._fallback_plate_from_image(
                width,
                height,
            )
        )

        return (
            "",
            plate,
            conf,
            status,
        )

    # =========================================================================
    # OCR
    # =========================================================================

    @classmethod
    def _ocr_full_image_with_raw(
        cls,
        image: Any,
    ) -> Tuple[str, str, float]:

        variants = cls._preprocess_variants(
            image
        )

        candidates = []

        # =====================================================================
        # EASY OCR
        # =====================================================================

        try:

            import easyocr
            import numpy as np

            if cls._ocr_reader is None:

                cls._ocr_reader = easyocr.Reader(
                    ["en"],
                    gpu=False,
                )

            for variant in variants:

                results = cls._ocr_reader.readtext(
                    np.array(variant),
                    detail=1,
                    paragraph=False,
                )

                if not results:
                    continue

                raw = " ".join(
                    text
                    for _, text, _ in results
                )

                plate = cls._extract_plate_from_text(
                    raw
                )

                if plate:

                    conf = max(
                        (
                            float(confidence)
                            for _, _, confidence
                            in results
                        ),
                        default=0.5,
                    )

                    score = cls._score_plate(
                        plate
                    )

                    candidates.append(
                        (
                            raw,
                            plate,
                            float(conf),
                            score,
                        )
                    )

        except Exception:
            pass

        # =====================================================================
        # TESSERACT
        # =====================================================================

        try:

            import pytesseract

            configs = [

                # Standard
                (
                    "--psm 6 --oem 3 "
                    "-c tessedit_char_whitelist="
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                ),

                # Sparse
                (
                    "--psm 11 --oem 3 "
                    "-c tessedit_char_whitelist="
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                ),

                # Automatic
                (
                    "--psm 3 --oem 3 "
                    "-c tessedit_char_whitelist="
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                ),

                # Single line
                (
                    "--psm 7 --oem 3 "
                    "-c tessedit_char_whitelist="
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                ),

                # Single word
                (
                    "--psm 8 --oem 3 "
                    "-c tessedit_char_whitelist="
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                ),

                # Raw line
                (
                    "--psm 13 --oem 3 "
                    "-c tessedit_char_whitelist="
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                ),
            ]

            for variant in variants:

                for config in configs:

                    try:

                        raw = pytesseract.image_to_string(
                            variant,
                            config=config,
                        )

                        raw = raw.strip()

                        if not raw:
                            continue

                        plate = (
                            cls._extract_plate_from_text(
                                raw
                            )
                        )

                        if not plate:
                            continue

                        score = cls._score_plate(
                            plate
                        )

                        candidates.append(
                            (
                                raw,
                                plate,
                                0.70,
                                score,
                            )
                        )

                    except Exception:
                        continue

        except Exception:
            pass

        # =====================================================================
        # SELECT BEST CANDIDATE
        # =====================================================================

        if not candidates:

            return "", "", 0.0

        # Count repeated candidates.
        #
        # If several OCR variants independently produce the same plate,
        # increase its score.

        frequency = {}

        for (
            raw,
            plate,
            confidence,
            score,
        ) in candidates:

            frequency[plate] = (
                frequency.get(plate, 0) + 1
            )

        ranked = []

        for candidate in candidates:

            raw, plate, confidence, score = (
                candidate
            )

            repetition_bonus = (
                frequency.get(plate, 1) - 1
            ) * 0.5

            final_score = (
                score
                + repetition_bonus
                + confidence
            )

            ranked.append(
                (
                    final_score,
                    raw,
                    plate,
                    confidence,
                    score,
                )
            )

        ranked.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        (
            _,
            best_raw,
            best_plate,
            best_confidence,
            _,
        ) = ranked[0]

        return (
            best_raw,
            best_plate,
            best_confidence,
        )

    # =========================================================================
    # PLATE-FOCUSED OCR
    # =========================================================================

    @classmethod
    def _plate_focused_ocr(
        cls,
        image: Any,
    ) -> Tuple[str, float, str]:

        """
        Special OCR pass for Nigerian plates.

        Nigerian plates commonly use red characters.

        This pass isolates the red channel using OpenCV/HSV.

        For the supplied FKJ-254-XA sample this type of preprocessing
        is significantly more useful than grayscale OCR.
        """

        try:

            import cv2
            import numpy as np
            import pytesseract

        except ImportError:

            return "", 0.0, ""

        try:

            img = np.array(
                image.convert("RGB")
            )

            # RGB → BGR
            bgr = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2BGR,
            )

            hsv = cv2.cvtColor(
                bgr,
                cv2.COLOR_BGR2HSV,
            )

            masks = []

            # -----------------------------------------------------------------
            # Red range 1
            # -----------------------------------------------------------------

            masks.append(
                cv2.inRange(
                    hsv,
                    np.array([0, 50, 50]),
                    np.array([15, 255, 255]),
                )
            )

            # -----------------------------------------------------------------
            # Red range 2
            # -----------------------------------------------------------------

            masks.append(
                cv2.inRange(
                    hsv,
                    np.array([0, 100, 100]),
                    np.array([10, 255, 255]),
                )
            )

            # -----------------------------------------------------------------
            # Red range 3
            # -----------------------------------------------------------------

            masks.append(
                cv2.inRange(
                    hsv,
                    np.array([170, 50, 50]),
                    np.array([180, 255, 255]),
                )
            )

            candidates = []

            configs = [
                "--psm 7 --oem 3",
                "--psm 8 --oem 3",
                "--psm 13 --oem 3",
                "--psm 6 --oem 3",
            ]

            for mask in masks:

                # Upscale heavily.
                enlarged = cv2.resize(
                    mask,
                    None,
                    fx=5,
                    fy=5,
                    interpolation=cv2.INTER_CUBIC,
                )

                # Clean small noise.
                kernel = np.ones(
                    (2, 2),
                    np.uint8,
                )

                enlarged = cv2.morphologyEx(
                    enlarged,
                    cv2.MORPH_CLOSE,
                    kernel,
                )

                for config in configs:

                    try:

                        raw = pytesseract.image_to_string(
                            enlarged,
                            config=(
                                config
                                + " "
                                "-c tessedit_char_whitelist="
                                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                            ),
                        )

                        raw = raw.strip()

                        if not raw:
                            continue

                        plate = (
                            cls._extract_plate_from_text(
                                raw
                            )
                        )

                        if not plate:
                            continue

                        score = cls._score_plate(
                            plate
                        )

                        candidates.append(
                            (
                                score,
                                raw,
                                plate,
                            )
                        )

                    except Exception:
                        continue

            if not candidates:
                return "", 0.0, ""

            candidates.sort(
                key=lambda x: x[0],
                reverse=True,
            )

            best_score, best_raw, best_plate = (
                candidates[0]
            )

            confidence = min(
                0.95,
                0.55 + (
                    best_score * 0.03
                ),
            )

            return (
                best_plate,
                confidence,
                best_raw,
            )

        except Exception:

            return "", 0.0, ""

    # =========================================================================
    # PREPROCESSING
    # =========================================================================

    @staticmethod
    def _preprocess_variants(
        image: Any,
    ) -> List[Any]:

        variants = []

        gray = image.convert("L")

        w, h = image.size

        r, g, b = (
            image.convert("RGB").split()
        )

        # ---------------------------------------------------------------------
        # Basic grayscale
        # ---------------------------------------------------------------------

        gray_2x = gray.resize(
            (w * 2, h * 2),
            Image.LANCZOS,
        )

        gray_3x = gray.resize(
            (w * 3, h * 3),
            Image.LANCZOS,
        )

        variants.append(gray_2x)
        variants.append(gray_3x)

        # ---------------------------------------------------------------------
        # Contrast
        # ---------------------------------------------------------------------

        variants.append(
            ImageOps.autocontrast(
                gray_2x,
                cutoff=2,
            )
        )

        variants.append(
            ImageOps.autocontrast(
                gray,
                cutoff=2,
            ).filter(
                ImageFilter.SHARPEN
            )
        )

        variants.append(
            ImageEnhance.Contrast(
                gray
            ).enhance(
                2.5
            ).filter(
                ImageFilter.SHARPEN
            )
        )

        # ---------------------------------------------------------------------
        # Red channel
        # ---------------------------------------------------------------------

        red_2x = r.resize(
            (w * 2, h * 2),
            Image.LANCZOS,
        )

        variants.append(red_2x)

        # ---------------------------------------------------------------------
        # Blue channel inverted
        # ---------------------------------------------------------------------

        inv_blue = ImageOps.invert(
            b
        ).resize(
            (w * 2, h * 2),
            Image.LANCZOS,
        )

        variants.append(inv_blue)

        # ---------------------------------------------------------------------
        # OpenCV preprocessing
        # ---------------------------------------------------------------------

        try:

            import cv2
            import numpy as np

            rgb = np.array(
                image.convert("RGB")
            )

            bgr = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR,
            )

            hsv = cv2.cvtColor(
                bgr,
                cv2.COLOR_BGR2HSV,
            )

            # Red mask
            red_mask = cv2.inRange(
                hsv,
                np.array([0, 50, 50]),
                np.array([15, 255, 255]),
            )

            red_mask = cv2.resize(
                red_mask,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC,
            )

            variants.append(
                Image.fromarray(
                    red_mask
                )
            )

            # Stronger red mask
            strong_red = cv2.inRange(
                hsv,
                np.array([0, 100, 100]),
                np.array([10, 255, 255]),
            )

            strong_red = cv2.resize(
                strong_red,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC,
            )

            variants.append(
                Image.fromarray(
                    strong_red
                )
            )

        except Exception:
            pass

        return variants

    # =========================================================================
    # TEXT → PLATE EXTRACTION
    # =========================================================================

    @classmethod
    def _extract_plate_from_text(
        cls,
        text: str,
    ) -> str:

        if not text:
            return ""

        upper = text.upper()

        # ---------------------------------------------------------------------
        # Strategy 1
        #
        # Exact Nigerian pattern.
        #
        # ABC123DE
        # ---------------------------------------------------------------------

        match = cls._PLATE_RE.search(
            upper
        )

        if match:

            candidate = re.sub(
                r"[^A-Z0-9]",
                "",
                match.group(1),
            )

            corrected = (
                cls._correct_plate(
                    candidate
                )
            )

            if cls._is_valid_nigerian_plate(
                corrected
            ):
                return corrected

        # ---------------------------------------------------------------------
        # Strategy 2
        #
        # Remove known noise words.
        # ---------------------------------------------------------------------

        cleaned = upper

        noise_words = (
            "FEDERAL REPUBLIC OF NIGERIA",
            "CENTRE OF EXCELLENCE",
            "REPUBLIC OF NIGERIA",
            "NIGERIA",
            "FEDERAL",
            "LAGOS",
            "ABUJA",
            "KADUNA",
            "KANO",
            "RIVERS",
            "OGUN",
            "OYO",
            "ANAMBRA",
            "ENUGU",
            "DELTA",
            "ONDO",
            "BENUE",
            "KOGI",
            "PLATEAU",
            "BAUCHI",
            "GOMBE",
            "YOBE",
            "BORNO",
            "ADAMAWA",
            "TARABA",
            "NASARAWA",
            "NIGER",
            "KWARA",
            "EKITI",
            "OSUN",
            "EDO",
            "IMO",
            "ABIA",
            "EBONYI",
            "BAYELSA",
            "FRSC",
            "VIO",
            "GOVERNMENT",
            "CENTRE",
            "EXCELLENCE",
            "UNITY",
        )

        for noise in noise_words:

            cleaned = cleaned.replace(
                noise,
                " ",
            )

        # ---------------------------------------------------------------------
        # Tokenize
        # ---------------------------------------------------------------------

        tokens = [
            re.sub(
                r"[^A-Z0-9]",
                "",
                token,
            )
            for token in re.split(
                r"[\s\-_/\\|,.]+",
                cleaned,
            )
        ]

        tokens = [
            token
            for token in tokens
            if token
        ]

        # ---------------------------------------------------------------------
        # Try individual tokens
        # ---------------------------------------------------------------------

        for token in tokens:

            corrected = (
                cls._correct_plate(
                    token
                )
            )

            if cls._is_valid_nigerian_plate(
                corrected
            ):
                return corrected

        # ---------------------------------------------------------------------
        # Try joining two tokens
        # ---------------------------------------------------------------------

        for i in range(
            len(tokens) - 1
        ):

            joined = (
                tokens[i]
                + tokens[i + 1]
            )

            corrected = (
                cls._correct_plate(
                    joined
                )
            )

            if cls._is_valid_nigerian_plate(
                corrected
            ):
                return corrected

        # ---------------------------------------------------------------------
        # Try joining three tokens
        # ---------------------------------------------------------------------

        for i in range(
            len(tokens) - 2
        ):

            joined = (
                tokens[i]
                + tokens[i + 1]
                + tokens[i + 2]
            )

            corrected = (
                cls._correct_plate(
                    joined
                )
            )

            if cls._is_valid_nigerian_plate(
                corrected
            ):
                return corrected

        # ---------------------------------------------------------------------
        # Try all loose alphanumeric chunks.
        # ---------------------------------------------------------------------

        loose_candidates = (
            cls._LOOSE_PLATE_RE.findall(
                upper
            )
        )

        for candidate in loose_candidates:

            corrected = (
                cls._correct_plate(
                    candidate
                )
            )

            if cls._is_valid_nigerian_plate(
                corrected
            ):
                return corrected

        return ""

    # =========================================================================
    # PLATE CORRECTION
    # =========================================================================

    @classmethod
    def _correct_plate(
        cls,
        plate: str,
    ) -> str:

        """
        Correct OCR output to Nigerian format:

            ABC123DE

        Example:

            FAJ254XA
                ↓
            FKJ254XA

        because FKJ is a known Nigerian/Lagos prefix.
        """

        if not plate:
            return ""

        p = re.sub(
            r"[^A-Z0-9]",
            "",
            plate.upper(),
        )

        if not p:
            return ""

        # =====================================================================
        # CASE 1
        #
        # Already exactly 8 characters.
        # =====================================================================

        if len(p) == 8:

            prefix = p[:3]
            middle = p[3:6]
            suffix = p[6:8]

            # ---------------------------------------------------------------
            # Correct prefix using known Nigerian prefixes
            # ---------------------------------------------------------------

            prefix = (
                cls._correct_prefix(
                    prefix
                )
            )

            # ---------------------------------------------------------------
            # Middle must be numbers
            # ---------------------------------------------------------------

            middle = "".join(
                DIGIT_FIXES.get(
                    char,
                    char,
                )
                for char in middle
            )

            # ---------------------------------------------------------------
            # Suffix must be letters
            # ---------------------------------------------------------------

            suffix = "".join(
                LETTER_FIXES.get(
                    char,
                    char,
                )
                for char in suffix
            )

            candidate = (
                prefix
                + middle
                + suffix
            )

            if cls._is_valid_nigerian_plate(
                candidate
            ):
                return candidate

        # =====================================================================
        # CASE 2
        #
        # OCR might have separators:
        #
        # FKJ-254-XA
        # =====================================================================

        compact = re.sub(
            r"[^A-Z0-9]",
            "",
            p,
        )

        if len(compact) == 8:

            prefix = cls._correct_prefix(
                compact[:3]
            )

            middle = "".join(
                DIGIT_FIXES.get(
                    char,
                    char,
                )
                for char in compact[3:6]
            )

            suffix = "".join(
                LETTER_FIXES.get(
                    char,
                    char,
                )
                for char in compact[6:8]
            )

            candidate = (
                prefix
                + middle
                + suffix
            )

            if cls._is_valid_nigerian_plate(
                candidate
            ):
                return candidate

        # =====================================================================
        # CASE 3
        #
        # Generic correction.
        # =====================================================================

        segments = re.split(
            r"(\d+)",
            p,
        )

        output = []

        for segment in segments:

            if not segment:
                continue

            if segment[0].isdigit():

                output.append(
                    "".join(
                        DIGIT_FIXES.get(
                            char,
                            char,
                        )
                        for char in segment
                    )
                )

            else:

                output.append(
                    "".join(
                        LETTER_FIXES.get(
                            char,
                            char,
                        )
                        for char in segment
                    )
                )

        generic = "".join(
            output
        )

        # If generic correction happens to create a valid plate,
        # apply prefix correction again.

        if len(generic) == 8:

            prefix = cls._correct_prefix(
                generic[:3]
            )

            middle = "".join(
                DIGIT_FIXES.get(
                    char,
                    char,
                )
                for char in generic[3:6]
            )

            suffix = "".join(
                LETTER_FIXES.get(
                    char,
                    char,
                )
                for char in generic[6:8]
            )

            candidate = (
                prefix
                + middle
                + suffix
            )

            if cls._is_valid_nigerian_plate(
                candidate
            ):
                return candidate

        return generic

    # =========================================================================
    # PREFIX CORRECTION
    # =========================================================================

    @classmethod
    def _correct_prefix(
        cls,
        prefix: str,
    ) -> str:

        prefix = prefix.upper()

        # Exact known prefix
        if prefix in PREFIX_TO_STATE:
            return prefix

        # ---------------------------------------------------------------------
        # Character-level OCR corrections
        # ---------------------------------------------------------------------

        corrected = "".join(
            LETTER_FIXES.get(
                char,
                char,
            )
            for char in prefix
        )

        if corrected in PREFIX_TO_STATE:
            return corrected

        # ---------------------------------------------------------------------
        # Fuzzy matching against known Nigerian prefixes.
        #
        # Example:
        #
        # FAJ
        #  ↓
        # FKJ
        #
        # Both are only one character apart.
        # ---------------------------------------------------------------------

        best_prefix = prefix
        best_distance = 999

        for known_prefix in PREFIX_TO_STATE:

            if len(known_prefix) != len(prefix):
                continue

            distance = (
                cls._levenshtein_distance(
                    prefix,
                    known_prefix,
                )
            )

            if distance < best_distance:

                best_distance = distance
                best_prefix = known_prefix

        # Allow only small OCR errors.
        if best_distance <= 1:

            return best_prefix

        return corrected

    # =========================================================================
    # LEVENSHTEIN DISTANCE
    # =========================================================================

    @staticmethod
    def _levenshtein_distance(
        a: str,
        b: str,
    ) -> int:

        if a == b:
            return 0

        if not a:
            return len(b)

        if not b:
            return len(a)

        previous = list(
            range(
                len(b) + 1
            )
        )

        for i, char_a in enumerate(
            a,
            start=1,
        ):

            current = [i]

            for j, char_b in enumerate(
                b,
                start=1,
            ):

                insert_cost = (
                    current[j - 1] + 1
                )

                delete_cost = (
                    previous[j] + 1
                )

                replace_cost = (
                    previous[j - 1]
                    + (
                        0
                        if char_a == char_b
                        else 1
                    )
                )

                current.append(
                    min(
                        insert_cost,
                        delete_cost,
                        replace_cost,
                    )
                )

            previous = current

        return previous[-1]

    # =========================================================================
    # VALIDATE NIGERIAN PLATE
    # =========================================================================

    @staticmethod
    def _is_valid_nigerian_plate(
        plate: str,
    ) -> bool:

        if not plate:
            return False

        return bool(
            re.fullmatch(
                r"[A-Z]{3}\d{3}[A-Z]{2}",
                plate,
            )
        )

    # =========================================================================
    # PLATE SCORING
    # =========================================================================

    @classmethod
    def _score_plate(
        cls,
        plate: str,
    ) -> int:

        if not plate:
            return 0

        # Exact Nigerian structure
        if cls._is_valid_nigerian_plate(
            plate
        ):

            score = 10

            prefix = plate[:3]

            # Known Nigerian prefix
            if prefix in PREFIX_TO_STATE:
                score += 5

            return score

        # Almost valid structure
        if (
            re.match(
                r"^[A-Z]{3}",
                plate,
            )
            and re.search(
                r"\d{2,4}",
                plate,
            )
            and re.search(
                r"[A-Z]{1,3}$",
                plate,
            )
        ):

            return 4

        # Generic alphanumeric candidate
        if (
            re.search(
                r"[A-Z]",
                plate,
            )
            and re.search(
                r"[0-9]",
                plate,
            )
        ):

            return 1

        return 0

    # =========================================================================
    # PLATE INFORMATION
    # =========================================================================

    @classmethod
    def _extract_plate_info(
        cls,
        raw_text: str,
        plate_number: str,
    ) -> Dict[str, str]:

        upper = (
            raw_text or ""
        ).upper()

        country = None
        state = None
        slogan = None

        # ---------------------------------------------------------------------
        # Country
        # ---------------------------------------------------------------------

        if (
            "NIGERIA" in upper
            or "FEDERAL REPUBLIC" in upper
        ):

            country = (
                "Federal Republic of Nigeria"
            )

        # ---------------------------------------------------------------------
        # State from OCR text
        # ---------------------------------------------------------------------

        for (
            keyword,
            (
                state_name,
                state_slogan,
            ),
        ) in NIGERIAN_STATES.items():

            if keyword in upper:

                state = state_name
                slogan = state_slogan

                break

        # ---------------------------------------------------------------------
        # State from plate prefix
        # ---------------------------------------------------------------------

        if not state and plate_number:

            prefix = (
                plate_number[:3]
                .upper()
            )

            state_key = (
                PREFIX_TO_STATE.get(
                    prefix
                )
            )

            if state_key:

                state, slogan = (
                    NIGERIAN_STATES.get(
                        state_key,
                        (
                            None,
                            None,
                        ),
                    )
                )

        # ---------------------------------------------------------------------
        # Always identify Nigerian plates
        # ---------------------------------------------------------------------

        if not country:

            country = (
                "Federal Republic of Nigeria"
            )

        # ---------------------------------------------------------------------
        # Clean raw OCR text
        # ---------------------------------------------------------------------

        cleaned_lines = []

        if raw_text:

            cleaned_lines = [
                line.strip()
                for line in raw_text.splitlines()
                if (
                    line.strip()
                    and len(line.strip()) > 1
                )
            ]

        clean = " | ".join(
            cleaned_lines
        )

        return {
            "country": country,
            "state": state,
            "slogan": slogan,
            "raw_text": clean,
        }

    # =========================================================================
    # YOLO PLATE DETECTION
    # =========================================================================

    @classmethod
    def _detect_plate_region(
        cls,
        image: Any,
        model_name: str,
    ) -> Optional[
        Tuple[str, str, float]
    ]:

        try:
            import numpy as np

        except ImportError:

            return None

        try:
            from ultralytics import YOLO

        except ImportError:

            return None

        detector = cls._load_detector(
            model_name
        )

        if detector is None:
            return None

        try:

            results = detector(
                np.array(image),
                imgsz=640,
                conf=0.25,
                stream=False,
            )[0]

        except Exception:

            return None

        detected_boxes = []

        for box in getattr(
            results,
            "boxes",
            [],
        ):

            try:

                conf = float(
                    box.conf[0]
                )

            except Exception:

                conf = 0.0

            if conf < 0.20:
                continue

            names = getattr(
                results,
                "names",
                {},
            )

            try:

                class_id = int(
                    box.cls[0]
                )

                if isinstance(
                    names,
                    dict,
                ):

                    label_name = names.get(
                        class_id,
                        "",
                    )

                else:

                    label_name = names[
                        class_id
                    ]

            except Exception:

                label_name = ""

            label_name = (
                label_name or ""
            ).lower()

            # IMPORTANT:
            #
            # Do NOT treat a normal "car" detection
            # as a license plate.
            #
            # A car bounding box is much larger than
            # the actual plate and gives bad OCR crops.

            if any(
                keyword in label_name
                for keyword in (
                    "plate",
                    "license plate",
                    "number plate",
                    "licence plate",
                )
            ):

                detected_boxes.append(
                    (
                        box,
                        conf,
                    )
                )

        if not detected_boxes:
            return None

        # ---------------------------------------------------------------------
        # Use highest-confidence plate
        # ---------------------------------------------------------------------

        top_box, top_conf = max(
            detected_boxes,
            key=lambda item: item[1],
        )

        try:

            x1, y1, x2, y2 = map(
                int,
                top_box.xyxy[0].tolist(),
            )

        except Exception:

            return None

        width, height = (
            image.size
        )

        # ---------------------------------------------------------------------
        # Padding
        # ---------------------------------------------------------------------

        padx = max(
            5,
            int(
                (x2 - x1) * 0.10
            ),
        )

        pady = max(
            5,
            int(
                (y2 - y1) * 0.10
            ),
        )

        x1 = max(
            0,
            x1 - padx,
        )

        y1 = max(
            0,
            y1 - pady,
        )

        x2 = min(
            width,
            x2 + padx,
        )

        y2 = min(
            height,
            y2 + pady,
        )

        cropped = image.crop(
            (
                x1,
                y1,
                x2,
                y2,
            )
        )

        if (
            cropped.size[0] <= 0
            or cropped.size[1] <= 0
        ):

            return None

        # ---------------------------------------------------------------------
        # OCR cropped plate
        # ---------------------------------------------------------------------

        raw_text, plate_text, ocr_conf = (
            cls._ocr_full_image_with_raw(
                cropped
            )
        )

        if plate_text:

            return (
                raw_text,
                plate_text,
                round(
                    max(
                        top_conf,
                        ocr_conf,
                    ),
                    3,
                ),
            )

        # Try dedicated plate preprocessing
        plate_text, confidence, raw = (
            cls._plate_focused_ocr(
                cropped
            )
        )

        if plate_text:

            return (
                raw,
                plate_text,
                round(
                    max(
                        top_conf,
                        confidence,
                    ),
                    3,
                ),
            )

        return None

    # =========================================================================
    # LOAD YOLO MODEL
    # =========================================================================

    @classmethod
    def _load_detector(
        cls,
        model_name: str,
    ):

        # Don't accidentally reuse YOLOv8 for another selected model.
        if (
            cls._detector is not None
            and cls._detector_model_name
            == model_name
        ):

            return cls._detector

        # ---------------------------------------------------------------------
        # These models are not loaded through Ultralytics YOLO.
        #
        # Keep them available in the UI, but fall back to OCR until their
        # proper detector implementation is added.
        # ---------------------------------------------------------------------

        if model_name in (
            "Faster R-CNN",
            "SSD",
        ):

            return None

        file_map = {
            "YOLOv8": "yolov8n.pt",
            "YOLOv9": "yolov9c.pt",
            "YOLOv10": "yolov10n.pt",
        }

        model_file = file_map.get(
            model_name
        )

        if not model_file:
            return None

        try:

            from ultralytics import YOLO

            cls._detector = YOLO(
                model_file
            )

            cls._detector_model_name = (
                model_name
            )

            return cls._detector

        except Exception:

            cls._detector = None
            cls._detector_model_name = None

            return None

    # =========================================================================
    # DATABASE LOOKUP
    # =========================================================================

    @classmethod
    def _lookup_registered_vehicle(
        cls,
        plate_number: str,
    ) -> Optional[
        Dict[str, Any]
    ]:

        if not plate_number:
            return None

        normalized = (
            plate_number
            .strip()
            .upper()
        )

        # ---------------------------------------------------------------------
        # First try exact lookup
        # ---------------------------------------------------------------------

        record = (
            CarRegisteration.objects
            .filter(
                plate_number__iexact=normalized
            )
            .first()
        )

        # ---------------------------------------------------------------------
        # Backward-compatible fallback
        # ---------------------------------------------------------------------

        if record is None:

            record = (
                CarRegisteration.objects
                .filter(
                    plate_number__icontains=normalized
                )
                .first()
            )

        if record is None:
            return None

        return {
            "owner": str(
                record.owner.full_name
            ),
            "vehicle_type": (
                record.vehicle_type
            ),
            "vehicle_model": (
                record.model
            ),
            "vehicle_color": (
                record.color
            ),
            "registration_number": (
                record.plate_number
            ),
        }

    # =========================================================================
    # OCR AVAILABILITY
    # =========================================================================

    @staticmethod
    def _ocr_available() -> bool:

        # ---------------------------------------------------------------------
        # Tesseract
        # ---------------------------------------------------------------------

        try:

            import pytesseract
            from PIL import Image as PILImage

            pytesseract.image_to_string(
                PILImage.new(
                    "RGB",
                    (10, 10),
                    color=255,
                )
            )

            return True

        except Exception:
            pass

        # ---------------------------------------------------------------------
        # EasyOCR / OpenCV
        # ---------------------------------------------------------------------

        for library in (
            "easyocr",
            "cv2",
        ):

            try:

                __import__(
                    library
                )

                return True

            except ImportError:

                continue

        return False

    # =========================================================================
    # FALLBACK
    # =========================================================================

    @staticmethod
    def _fallback_plate_from_image(
        width: int,
        height: int,
    ) -> Tuple[
        str,
        float,
        str,
    ]:

        """
        Emergency fallback.

        IMPORTANT:
        This should not be considered a real OCR result.
        """

        prefix = "AB"

        middle = (
            (width * 3 + height)
            % 900
        ) + 100

        suffix = chr(
            65
            + (
                (width + height)
                % 8
            )
        )

        return (
            f"{prefix}{middle}{suffix}",
            0.35,
            "fallback",
        )