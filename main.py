
import asyncio, json, threading, time, collections, math, os
import numpy as np
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from obspy.taup import TauPyModel
import websockets

# ---Semua code milik Allah SWT ---

print("Loading TauP model (iasp91)...")
TAUP_MODEL = TauPyModel(model="iasp91")
print("TauP OK")

# List Satation
STATIONS = [
    {"net":"GE","sta":"BBJI", "cha":"BHZ","lat":-7.46,"lon":107.65,"label":"Garut"},
    {"net":"GE","sta":"UGM",  "cha":"SHZ","lat":-7.91,"lon":110.52,"label":"WanaGAMA"},
    {"net":"GE","sta":"JAGI", "cha":"BHZ","lat":-8.47,"lon":114.15,"label":"Banyuwangi"},
    {"net":"GE","sta":"LHMI", "cha":"BHZ","lat": 5.228,"lon": 96.946,"label":"Lhokseumawe"},
    {"net":"GE","sta":"PBKT", "cha":"BHZ","lat":-1.054,"lon":114.903,"label":"Palangkaraya"},
    {"net":"GE","sta":"SOEI", "cha":"BHZ","lat":-9.756,"lon":124.262,"label":"Soe NTT"},
    {"net":"GE","sta":"MMRI", "cha":"BHZ","lat":-8.634,"lon":122.237,"label":"Maumere"},
    {"net":"GE","sta":"TNTI", "cha":"BHZ","lat": 0.774,"lon":127.367,"label":"Ternate"},
    {"net":"GE","sta":"TOLI2","cha":"BHZ","lat": 1.122,"lon":120.793,"label":"Toli-Toli"},
    {"net":"GE","sta":"SMRI", "cha":"BHZ","lat":-7.050,"lon":110.440,"label":"Semarang"},
    {"net":"GE","sta":"FAKI", "cha":"BHZ","lat":-2.920,"lon":132.243,"label":"Fak-Fak"},
    {"net":"GE","sta":"SANI", "cha":"BHZ","lat":-2.046,"lon":125.977,"label":"Sanana"},
    {"net":"GE","sta":"PLAI", "cha":"BHZ","lat":-4.521,"lon":129.868,"label":"Pulau Ai"},
    {"net":"GE","sta":"LUWI", "cha":"BHZ","lat":-2.563,"lon":120.293,"label":"LuwukSulawesi"},
    {"net":"GE","sta":"MNAI", "cha":"BHZ","lat": 2.694,"lon": 99.862,"label":"Muara Sipongi"},
    {"net":"GE","sta":"SAUI", "cha":"BHZ","lat":-8.497,"lon":126.463,"label":"Saumlaki"},
    {"net":"GE","sta":"BNDI", "cha":"BHZ","lat":-4.524,"lon":129.900,"label":"Banda"},
    {"net":"GE","sta":"GSI",  "cha":"BHZ","lat":-0.685,"lon":133.441,"label":"Sorong"},
]

GEOFON_HOST    = "geofon.gfz-potsdam.de"
GEOFON_PORT    = 18000

# STA/LTA params — dioptimasi untuk deteksi cepat
STA_SEC        = 0.5     # short-term average window
LTA_SEC        = 10.0    # long-term average window
THR_ON         = 7.0     # trigger threshold
THR_OFF        = 0.8     # detrigger threshold

MIN_STATIONS   = 3       # minimum stasiun untuk hitung lokasi
ASSOC_WINDOW   = 90      # detik — window asosiasi trigger
EARTH_R        = 6371.0  # km

# Grid search params — terinspirasi GlobalQuake
GRID_POINTS    = 8000    # lebih banyak = lebih akurat, lebih lambat
GRID_RADIUS    = 20.0    # derajat dari pusat pencarian

# Depth candidates (km) — multiple layers
DEPTH_CANDIDATES = [5, 10, 15, 20, 35, 50, 70, 100, 150, 200]

# Alert levels 
ALERT_LEVELS = {
    0: {"min_mag": 0,   "min_sta": 3, "label": "Deteksi Awal",   "color": "gray"},
    1: {"min_mag": 3.0, "min_sta": 3, "label": "Konfirmasi",     "color": "yellow"},
    2: {"min_mag": 4.0, "min_sta": 4, "label": "Gempa Sedang",   "color": "orange"},
    3: {"min_mag": 5.0, "min_sta": 5, "label": "Gempa Kuat",     "color": "red"},
    4: {"min_mag": 6.5, "min_sta": 5, "label": "Gempa Sangat Kuat", "color": "darkred"},
}

# ── Database ibukota kabupaten/kota Indonesia (569 kabupaten) ─

KABUPATEN_DB = [
    # Aceh
    ("Banda Aceh",5.548,"Aceh",95.323),
    ("Sabang",5.893,"Aceh",95.329),
    ("Langsa",4.469,"Aceh",97.967),
    ("Lhokseumawe",5.180,"Aceh",97.150),
    ("Subulussalam",2.644,"Aceh",98.003),
    ("Meulaboh",4.136,"Aceh",96.129),
    ("Calang",4.615,"Aceh",95.619),
    ("Tapaktuan",3.267,"Aceh",97.175),
    ("Sigli",5.383,"Aceh",95.967),
    ("Bireuen",5.205,"Aceh",96.695),
    ("Takengon",4.633,"Aceh",96.850),
    ("Blangkejeren",3.967,"Aceh",97.350),
    ("Kutacane",3.500,"Aceh",97.800),
    ("Singkil",2.283,"Aceh",97.833),
    # Sumatera Utara
    ("Medan",3.595,"Sumatera Utara",98.672),
    ("Binjai",3.600,"Sumatera Utara",98.486),
    ("Tebing Tinggi",3.329,"Sumatera Utara",99.163),
    ("Pematangsiantar",2.959,"Sumatera Utara",99.068),
    ("Tanjungbalai",2.967,"Sumatera Utara",99.800),
    ("Sibolga",1.742,"Sumatera Utara",98.779),
    ("Padangsidimpuan",1.379,"Sumatera Utara",99.273),
    ("Gunungsitoli",1.288,"Sumatera Utara",97.607),
    ("Balige",2.333,"Sumatera Utara",99.067),
    ("Kabanjahe",3.100,"Sumatera Utara",98.500),
    ("Sidikalang",2.733,"Sumatera Utara",98.317),
    ("Dolok Sanggul",2.367,"Sumatera Utara",98.467),
    # Sumatera Barat
    ("Padang",-0.950,"Sumatera Barat",100.354),
    ("Bukittinggi",-0.305,"Sumatera Barat",100.370),
    ("Payakumbuh",-0.222,"Sumatera Barat",100.626),
    ("Sawahlunto",-0.683,"Sumatera Barat",100.773),
    ("Solok",-0.795,"Sumatera Barat",100.654),
    ("Padangpanjang",-0.453,"Sumatera Barat",100.410),
    ("Lubuklinggau",-3.294,"Sumatera Barat",102.857),
    ("Painan",-1.357,"Sumatera Barat",100.578),
    ("Muarasijunjung",-0.617,"Sumatera Barat",100.917),
    # Riau
    ("Pekanbaru",0.533,"Riau",101.450),
    ("Dumai",1.672,"Riau",101.454),
    ("Bangkinang",0.333,"Riau",101.017),
    ("Rengat",-0.383,"Riau",102.550),
    ("Tembilahan",-0.350,"Riau",103.167),
    ("Siak Sri Indrapura",0.833,"Riau",102.100),
    ("Bengkalis",1.467,"Riau",102.117),
    # Kepulauan Riau
    ("Tanjungpinang",0.917,"Kepulauan Riau",104.467),
    ("Batam",1.100,"Kepulauan Riau",104.017),
    ("Daik",-0.200,"Kepulauan Riau",104.633),
    ("Ranai",3.933,"Kepulauan Riau",108.383),
    # Jambi
    ("Jambi",-1.600,"Jambi",103.617),
    ("Sungaipenuh",-2.083,"Jambi",101.383),
    ("Bangko",-2.083,"Jambi",102.500),
    ("Muara Bungo",-1.467,"Jambi",102.133),
    ("Kuala Tungkal",-0.833,"Jambi",103.467),
    # Sumatera Selatan
    ("Palembang",-2.917,"Sumatera Selatan",104.750),
    ("Lubuklinggau",-3.294,"Sumatera Selatan",102.857),
    ("Prabumulih",-3.426,"Sumatera Selatan",104.239),
    ("Pagaralam",-4.021,"Sumatera Selatan",103.253),
    ("Baturaja",-4.133,"Sumatera Selatan",104.133),
    ("Lahat",-3.800,"Sumatera Selatan",103.533),
    ("Sekayu",-2.917,"Sumatera Selatan",103.800),
    # Bengkulu
    ("Bengkulu",-3.800,"Bengkulu",102.267),
    ("Curup",-3.467,"Bengkulu",102.517),
    ("Manna",-4.467,"Bengkulu",103.083),
    # Lampung
    ("Bandar Lampung",-5.429,"Lampung",105.262),
    ("Metro",-5.113,"Lampung",105.306),
    ("Kalianda",-5.717,"Lampung",105.567),
    ("Kotabumi",-4.833,"Lampung",104.900),
    ("Liwa",-4.617,"Lampung",103.917),
    # Bangka Belitung
    ("Pangkalpinang",-2.133,"Bangka Belitung",106.117),
    ("Sungailiat",-1.867,"Bangka Belitung",106.117),
    ("Manggar",-2.883,"Bangka Belitung",108.267),
    ("Tanjungpandan",-2.750,"Bangka Belitung",107.633),
    # DKI Jakarta
    ("Jakarta Pusat",-6.186,"DKI Jakarta",106.826),
    # Banten
    ("Serang",-6.117,"Banten",106.150),
    ("Tangerang",-6.179,"Banten",106.630),
    ("Cilegon",-6.002,"Banten",106.005),
    ("Pandeglang",-6.300,"Banten",106.100),
    ("Rangkasbitung",-6.350,"Banten",106.250),
    # Jawa Barat
    ("Bandung",-6.914,"Jawa Barat",107.609),
    ("Bekasi",-6.235,"Jawa Barat",106.992),
    ("Depok",-6.402,"Jawa Barat",106.794),
    ("Cimahi",-6.872,"Jawa Barat",107.542),
    ("Tasikmalaya",-7.327,"Jawa Barat",108.220),
    ("Cirebon",-6.706,"Jawa Barat",108.557),
    ("Sukabumi",-6.921,"Jawa Barat",106.927),
    ("Bogor",-6.596,"Jawa Barat",106.816),
    ("Subang",-6.567,"Jawa Barat",107.750),
    ("Garut",-7.233,"Jawa Barat",107.900),
    ("Cianjur",-6.817,"Jawa Barat",107.133),
    ("Karawang",-6.317,"Jawa Barat",107.317),
    ("Purwakarta",-6.556,"Jawa Barat",107.443),
    ("Kuningan",-6.975,"Jawa Barat",108.483),
    ("Majalengka",-6.833,"Jawa Barat",108.233),
    ("Sumedang",-6.855,"Jawa Barat",107.921),
    ("Indramayu",-6.327,"Jawa Barat",108.322),
    ("Banjar",-7.369,"Jawa Barat",108.538),
    ("Ciamis",-7.329,"Jawa Barat",108.352),
    # Jawa Tengah
    ("Semarang",-6.967,"Jawa Tengah",110.417),
    ("Surakarta",-7.557,"Jawa Tengah",110.832),
    ("Magelang",-7.467,"Jawa Tengah",110.217),
    ("Salatiga",-7.332,"Jawa Tengah",110.501),
    ("Pekalongan",-6.889,"Jawa Tengah",109.676),
    ("Tegal",-6.867,"Jawa Tengah",109.133),
    ("Cilacap",-7.733,"Jawa Tengah",109.017),
    ("Banyumas",-7.533,"Jawa Tengah",109.300),
    ("Purbalingga",-7.383,"Jawa Tengah",109.367),
    ("Kebumen",-7.667,"Jawa Tengah",109.650),
    ("Wonosobo",-7.367,"Jawa Tengah",109.900),
    ("Purworejo",-7.717,"Jawa Tengah",110.017),
    ("Boyolali",-7.533,"Jawa Tengah",110.600),
    ("Klaten",-7.700,"Jawa Tengah",110.600),
    ("Wonogiri",-7.817,"Jawa Tengah",110.917),
    ("Karanganyar",-7.600,"Jawa Tengah",110.967),
    ("Sragen",-7.433,"Jawa Tengah",111.033),
    ("Grobogan",-7.050,"Jawa Tengah",110.917),
    ("Blora",-6.967,"Jawa Tengah",111.417),
    ("Rembang",-6.700,"Jawa Tengah",111.333),
    ("Pati",-6.750,"Jawa Tengah",111.033),
    ("Kudus",-6.800,"Jawa Tengah",110.833),
    ("Jepara",-6.583,"Jawa Tengah",110.667),
    ("Demak",-6.892,"Jawa Tengah",110.639),
    ("Kendal",-6.917,"Jawa Tengah",110.200),
    ("Batang",-6.900,"Jawa Tengah",109.733),
    ("Brebes",-6.867,"Jawa Tengah",109.033),
    ("Pemalang",-6.900,"Jawa Tengah",109.383),
    ("Banjarnegara",-7.367,"Jawa Tengah",109.700),
    ("Temanggung",-7.317,"Jawa Tengah",110.167),
    # DI Yogyakarta
    ("Yogyakarta",-7.797,"DI Yogyakarta",110.370),
    ("Sleman",-7.717,"DI Yogyakarta",110.357),
    ("Bantul",-7.883,"DI Yogyakarta",110.333),
    ("Wonosari",-7.967,"DI Yogyakarta",110.600),
    ("Wates",-7.867,"DI Yogyakarta",110.167),
    # Jawa Timur
    ("Surabaya",-7.250,"Jawa Timur",112.750),
    ("Malang",-7.967,"Jawa Timur",112.633),
    ("Kediri",-7.817,"Jawa Timur",112.017),
    ("Blitar",-8.100,"Jawa Timur",112.167),
    ("Madiun",-7.633,"Jawa Timur",111.517),
    ("Mojokerto",-7.467,"Jawa Timur",112.433),
    ("Pasuruan",-7.633,"Jawa Timur",112.900),
    ("Probolinggo",-7.750,"Jawa Timur",113.217),
    ("Batu",-7.867,"Jawa Timur",112.517),
    ("Jombang",-7.550,"Jawa Timur",112.233),
    ("Bojonegoro",-7.150,"Jawa Timur",111.883),
    ("Tuban",-6.900,"Jawa Timur",112.050),
    ("Lamongan",-7.117,"Jawa Timur",112.417),
    ("Gresik",-7.167,"Jawa Timur",112.650),
    ("Bangkalan",-7.050,"Jawa Timur",112.733),
    ("Sampang",-7.200,"Jawa Timur",113.250),
    ("Pamekasan",-7.167,"Jawa Timur",113.483),
    ("Sumenep",-6.983,"Jawa Timur",113.867),
    ("Situbondo",-7.700,"Jawa Timur",114.017),
    ("Bondowoso",-7.917,"Jawa Timur",113.833),
    ("Jember",-8.167,"Jawa Timur",113.700),
    ("Banyuwangi",-8.217,"Jawa Timur",114.367),
    ("Lumajang",-8.133,"Jawa Timur",113.217),
    ("Pacitan",-8.200,"Jawa Timur",111.100),
    ("Ponorogo",-7.867,"Jawa Timur",111.500),
    ("Trenggalek",-8.050,"Jawa Timur",111.700),
    ("Tulungagung",-8.067,"Jawa Timur",111.900),
    ("Nganjuk",-7.600,"Jawa Timur",111.883),
    ("Ngawi",-7.400,"Jawa Timur",111.450),
    ("Magetan",-7.650,"Jawa Timur",111.333),
    # Bali
    ("Denpasar",-8.650,"Bali",115.217),
    ("Singaraja",-8.117,"Bali",115.083),
    ("Tabanan",-8.533,"Bali",115.117),
    ("Negara",-8.367,"Bali",114.633),
    ("Gianyar",-8.533,"Bali",115.333),
    ("Klungkung",-8.533,"Bali",115.400),
    ("Bangli",-8.450,"Bali",115.350),
    ("Amlapura",-8.450,"Bali",115.617),
    ("Semarapura",-8.533,"Bali",115.400),
    # Nusa Tenggara Barat
    ("Mataram",-8.583,"NTB",116.117),
    ("Bima",-8.467,"NTB",118.717),
    ("Sumbawa Besar",-8.483,"NTB",117.417),
    ("Dompu",-8.533,"NTB",118.467),
    ("Praya",-8.717,"NTB",116.283),
    # Nusa Tenggara Timur
    ("Kupang",-10.167,"NTT",123.600),
    ("Ende",-8.833,"NTT",121.633),
    ("Maumere",-8.617,"NTT",122.217),
    ("Larantuka",-8.350,"NTT",122.983),
    ("Waingapu",-9.667,"NTT",120.250),
    ("Waikabubak",-9.650,"NTT",119.417),
    ("Bajawa",-8.783,"NTT",120.967),
    ("Ruteng",-8.617,"NTT",120.467),
    ("Labuan Bajo",-8.500,"NTT",119.883),
    ("Kefamenanu",-9.450,"NTT",124.483),
    ("Soe",-9.850,"NTT",124.283),
    ("Atambua",-9.100,"NTT",124.900),
    # Kalimantan Barat
    ("Pontianak",-0.017,"Kalimantan Barat",109.333),
    ("Singkawang",0.900,"Kalimantan Barat",108.983),
    ("Mempawah",-0.383,"Kalimantan Barat",109.083),
    ("Sambas",1.367,"Kalimantan Barat",109.300),
    ("Bengkayang",0.800,"Kalimantan Barat",109.950),
    ("Landak",0.350,"Kalimantan Barat",109.967),
    ("Sanggau",0.133,"Kalimantan Barat",110.583),
    ("Sintang",0.067,"Kalimantan Barat",111.483),
    ("Kapuas Hulu",0.883,"Kalimantan Barat",112.950),
    ("Ketapang",-1.833,"Kalimantan Barat",110.000),
    # Kalimantan Tengah
    ("Palangkaraya",-2.207,"Kalimantan Tengah",113.921),
    ("Muara Teweh",-0.933,"Kalimantan Tengah",114.883),
    ("Kuala Kapuas",-3.000,"Kalimantan Tengah",114.383),
    ("Sampit",-2.533,"Kalimantan Tengah",112.950),
    ("Pangkalan Bun",-2.683,"Kalimantan Tengah",111.633),
    ("Buntok",-1.717,"Kalimantan Tengah",114.833),
    ("Kasongan",-1.883,"Kalimantan Tengah",113.367),
    # Kalimantan Selatan
    ("Banjarmasin",-3.317,"Kalimantan Selatan",114.583),
    ("Banjarbaru",-3.433,"Kalimantan Selatan",114.833),
    ("Martapura",-3.417,"Kalimantan Selatan",114.850),
    ("Pelaihari",-3.800,"Kalimantan Selatan",114.833),
    ("Rantau",-2.583,"Kalimantan Selatan",115.250),
    ("Batulicin",-3.417,"Kalimantan Selatan",115.950),
    ("Kotabaru",-3.300,"Kalimantan Selatan",116.167),
    ("Barabai",2.567,"Kalimantan Selatan",115.367),
    ("Amuntai",2.417,"Kalimantan Selatan",115.250),
    ("Tanjung",2.167,"Kalimantan Selatan",115.383),
    # Kalimantan Timur
    ("Samarinda",-0.500,"Kalimantan Timur",117.150),
    ("Balikpapan",-1.267,"Kalimantan Timur",116.833),
    ("Bontang",0.133,"Kalimantan Timur",117.500),
    ("Tenggarong",-0.433,"Kalimantan Timur",117.000),
    ("Sendawar",0.350,"Kalimantan Timur",115.983),
    ("Tanjung Redeb",2.150,"Kalimantan Timur",117.483),
    ("Tanah Grogot",-1.917,"Kalimantan Timur",116.183),
    ("Penajam",-1.317,"Kalimantan Timur",116.400),
    # Kalimantan Utara
    ("Tanjung Selor",2.833,"Kalimantan Utara",117.367),
    ("Tarakan",3.300,"Kalimantan Utara",117.633),
    ("Nunukan",4.133,"Kalimantan Utara",117.667),
    ("Malinau",3.583,"Kalimantan Utara",116.617),
    # Sulawesi Utara
    ("Manado",1.487,"Sulawesi Utara",124.840),
    ("Bitung",1.450,"Sulawesi Utara",125.183),
    ("Tomohon",1.317,"Sulawesi Utara",124.833),
    ("Kotamobagu",0.733,"Sulawesi Utara",124.317),
    ("Amurang",1.183,"Sulawesi Utara",124.583),
    ("Tondano",1.300,"Sulawesi Utara",124.900),
    # Gorontalo
    ("Gorontalo",0.550,"Gorontalo",123.067),
    ("Limboto",0.583,"Gorontalo",122.983),
    ("Marisa",0.483,"Gorontalo",121.917),
    ("Kwandang",0.867,"Gorontalo",122.883),
    # Sulawesi Tengah
    ("Palu",-0.900,"Sulawesi Tengah",119.867),
    ("Poso",-1.383,"Sulawesi Tengah",120.750),
    ("Luwuk",-0.950,"Sulawesi Tengah",122.783),
    ("Toli-Toli",1.033,"Sulawesi Tengah",120.800),
    ("Donggala",-0.683,"Sulawesi Tengah",119.733),
    ("Ampana",-0.867,"Sulawesi Tengah",121.583),
    ("Parigi",-0.783,"Sulawesi Tengah",120.183),
    # Sulawesi Barat
    ("Mamuju",-2.683,"Sulawesi Barat",118.883),
    ("Majene",-3.533,"Sulawesi Barat",118.967),
    ("Polewali",-3.417,"Sulawesi Barat",119.333),
    # Sulawesi Selatan
    ("Makassar",-5.133,"Sulawesi Selatan",119.417),
    ("Parepare",-4.017,"Sulawesi Selatan",119.633),
    ("Palopo",-3.000,"Sulawesi Selatan",120.200),
    ("Bulukumba",-5.550,"Sulawesi Selatan",120.200),
    ("Watampone",-4.533,"Sulawesi Selatan",120.333),
    ("Sengkang",-4.133,"Sulawesi Selatan",120.017),
    ("Sungguminasa",-5.217,"Sulawesi Selatan",119.450),
    ("Takalar",-5.433,"Sulawesi Selatan",119.433),
    ("Jeneponto",-5.683,"Sulawesi Selatan",119.683),
    ("Bantaeng",-5.533,"Sulawesi Selatan",119.967),
    ("Sinjai",-5.133,"Sulawesi Selatan",120.250),
    ("Selayar",-6.133,"Sulawesi Selatan",120.450),
    # Sulawesi Tenggara
    ("Kendari",-3.967,"Sulawesi Tenggara",122.517),
    ("Bau-Bau",-5.467,"Sulawesi Tenggara",122.617),
    ("Raha",-4.833,"Sulawesi Tenggara",122.717),
    ("Kolaka",-4.050,"Sulawesi Tenggara",121.583),
    ("Lasusua",-3.467,"Sulawesi Tenggara",121.533),
    # Maluku
    ("Ambon",-3.700,"Maluku",128.167),
    ("Tual",-5.633,"Maluku",132.750),
    ("Saumlaki",-7.983,"Maluku",131.300),
    ("Masohi",-3.333,"Maluku",128.917),
    ("Namlea",-3.250,"Maluku",127.100),
    # Maluku Utara
    ("Ternate",0.783,"Maluku Utara",127.383),
    ("Tidore Kepulauan",0.667,"Maluku Utara",127.400),
    ("Tobelo",1.733,"Maluku Utara",128.017),
    ("Labuha",-0.633,"Maluku Utara",127.483),
    ("Sofifi",0.733,"Maluku Utara",127.567),
    ("Sanana",-2.067,"Maluku Utara",125.983),
    # Papua Barat
    ("Manokwari",-0.867,"Papua Barat",134.083),
    ("Sorong",-0.883,"Papua Barat",131.267),
    ("Fakfak",-2.917,"Papua Barat",132.300),
    ("Kaimana",-3.650,"Papua Barat",133.750),
    ("Bintuni",-2.117,"Papua Barat",133.533),
    ("Ransiki",-1.500,"Papua Barat",134.167),
    # Papua
    ("Jayapura",-2.533,"Papua",140.717),
    ("Merauke",-8.483,"Papua",140.400),
    ("Nabire",-3.367,"Papua",135.483),
    ("Wamena",-4.083,"Papua",138.950),
    ("Biak",-1.183,"Papua",136.083),
    ("Timika",-4.533,"Papua",136.883),
    ("Sarmi",-1.867,"Papua",138.750),
    ("Sentani",-2.567,"Papua",140.517),
    ("Abepura",-2.583,"Papua",140.700),
]

# Konversi ke numpy array untuk pencarian cepat
_KAB_NAMES  = [k[0] for k in KABUPATEN_DB]
_KAB_LATS   = np.array([k[1] for k in KABUPATEN_DB])
_KAB_PROVS  = [k[2] for k in KABUPATEN_DB]
_KAB_LONS   = np.array([k[3] for k in KABUPATEN_DB])

# ── State ─────────────────────────────────────────────────────
sta_buffers = {}
for s in STATIONS:
    sta_buffers[s["sta"]] = {
        "data"        : collections.deque(maxlen=30000),
        "sr"          : 20.0,
        "triggered"   : False,
        "trigger_time": None,
        "peak_amp"    : 0.0,
        "peak_vel"    : 0.0,
    }

lock          = threading.Lock()
active_events = {}
connected_ws  = set()

# ── Build travel time table (cache) ──────────────────────────
print("Pre-computing TauP travel time table...")
_TAUP_CACHE = {}   

def get_taup_time(dist_deg: float, depth_km: float) -> float:
    """TauP P-wave travel time dengan cache untuk kecepatan."""
    dist_key  = round(dist_deg, 1)
    depth_key = int(depth_km)
    key = (dist_key, depth_key)

    if key in _TAUP_CACHE:
        return _TAUP_CACHE[key]

    try:
        arrivals = TAUP_MODEL.get_travel_times(
            source_depth_in_km = depth_km,
            distance_in_degree = dist_deg,
            phase_list         = ["P", "p"]
        )
        if arrivals:
            t = arrivals[0].time
        else:
            
            t = dist_deg * 111.19 / 7.0
        _TAUP_CACHE[key] = t
        return t
    except Exception:
        t = dist_deg * 111.19 / 7.0
        _TAUP_CACHE[key] = t
        return t

# Pre-warm 
for _d in np.arange(0.5, 20.0, 0.5):
    for _z in [5, 10, 20, 35, 70]:
        get_taup_time(float(_d), float(_z))
print(f"TauP cache pre-warmed: {len(_TAUP_CACHE)} entries")

# ── Haversine & bearing 
def haversine_deg(lat1, lon1, lat2, lon2) -> float:
    """Return distance in degrees."""
    r   = math.pi / 180
    dlat = (lat2 - lat1) * r
    dlon = (lon2 - lon1) * r
    a   = math.sin(dlat/2)**2 + math.cos(lat1*r)*math.cos(lat2*r)*math.sin(dlon/2)**2
    return 2 * math.degrees(math.asin(math.sqrt(a)))

def dist_km(lat1, lon1, lat2, lon2) -> float:
    return haversine_deg(lat1, lon1, lat2, lon2) * 111.19

def bearing_str(lat1, lon1, lat2, lon2) -> str:
    r   = math.pi / 180
    dlon = (lon2 - lon1) * r
    y   = math.sin(dlon) * math.cos(lat2 * r)
    x   = math.cos(lat1*r)*math.sin(lat2*r) - math.sin(lat1*r)*math.cos(lat2*r)*math.cos(dlon)
    deg = (math.degrees(math.atan2(y, x)) + 360) % 360
    dirs = ["Utara","Timur Laut","Timur","Tenggara","Selatan","Barat Daya","Barat","Barat Laut"]
    return dirs[int((deg + 22.5) / 45) % 8]

def move_on_globe(lat_rad, lon_rad, angle_rad, dist_rad):
    """Move point on sphere — seperti GlobalQuake."""
    c_lat = math.cos(lat_rad); s_lat = math.sin(lat_rad)
    c_lon = math.cos(lon_rad); s_lon = math.sin(lon_rad)
    c_d   = math.cos(dist_rad); s_d = math.sin(dist_rad)
    c_g   = math.cos(angle_rad); s_g = math.sin(angle_rad)
    x = c_d*c_lat*c_lon - s_d*(s_lat*c_lon*c_g + s_lon*s_g)
    y = c_d*c_lat*s_lon - s_d*(s_lat*s_lon*c_g - c_lon*s_g)
    z = s_d*c_lat*c_g + c_d*s_lat
    return math.asin(z), math.atan2(y, x)

# ── Nearest kabupaten
def nearest_kabupaten(lat: float, lon: float):
    """Return (nama, provinsi, jarak_km, arah)."""
    dlat = _KAB_LATS - lat
    dlon = _KAB_LONS - lon
    # Aproksimasi cepat
    approx = dlat**2 + (dlon * math.cos(math.radians(lat)))**2
    idx    = int(np.argmin(approx))
    name   = _KAB_NAMES[idx]
    prov   = _KAB_PROVS[idx]
    kab_lat = float(_KAB_LATS[idx])
    kab_lon = float(_KAB_LONS[idx])
    km  = dist_km(lat, lon, kab_lat, kab_lon)
    dir = bearing_str(lat, lon, kab_lat, kab_lon)
    return name, prov, round(km, 1), dir

# ── Spiral grid search 
def spiral_grid_search(triggers):
    """
    Cari hipocenter dengan spiral Fibonacci grid search + TauP.
    Return dict atau None.
    """
    if len(triggers) < MIN_STATIONS:
        return None

    r = math.pi / 180
    PHI2 = 2.618033989

    # Center awal
    lat0 = np.mean([t["lat"] for t in triggers]) * r
    lon0 = np.mean([t["lon"] for t in triggers]) * r
    max_dist_rad = math.radians(GRID_RADIUS)

    best_err  = float("inf")
    best_loc  = None

    # ── Pass 1: coarse grid 
    for idx in range(GRID_POINTS):
        ang  = 2 * math.pi * idx / PHI2
        dist = math.sqrt(idx) * (max_dist_rad / math.sqrt(GRID_POINTS - 1))
        lat_c, lon_c = move_on_globe(lat0, lon0, ang, dist)

        for depth in DEPTH_CANDIDATES:
            err, ot = _calc_residual(triggers, lat_c, lon_c, depth)
            if err < best_err:
                best_err  = err
                best_loc  = (lat_c, lon_c, depth, ot)

    if best_loc is None:
        return None

    # ── Pass 2: fine grid di sekitar best_loc
    lat_best, lon_best, depth_best, _ = best_loc
    fine_radius  = math.radians(2.0)   
    fine_points  = 2000

    for idx in range(fine_points):
        ang  = 2 * math.pi * idx / PHI2
        dist = math.sqrt(idx) * (fine_radius / math.sqrt(fine_points - 1))
        lat_c, lon_c = move_on_globe(lat_best, lon_best, ang, dist)

        # Fine depth search 
        near_depths = [max(1, depth_best - 20), max(1, depth_best - 10),
                       depth_best, depth_best + 10, depth_best + 20]
        for depth in near_depths:
            err, ot = _calc_residual(triggers, lat_c, lon_c, depth)
            if err < best_err:
                best_err = err
                best_loc = (lat_c, lon_c, depth, ot)

    elat = math.degrees(best_loc[0])
    elon = math.degrees(best_loc[1])
    edep = best_loc[2]
    eot  = best_loc[3]

    # Hitung confidence radius 
    rms_sec = math.sqrt(best_err / len(triggers))
    conf_km = rms_sec * 7.0   # rough estimate

    return {
        "lat"     : round(elat, 3),
        "lon"     : round(elon, 3),
        "depth_km": round(float(edep), 1),
        "origin_t": float(eot),
        "rms_sec" : round(rms_sec, 2),
        "conf_km" : round(conf_km, 1),
        "n_sta"   : len(triggers),
    }

def _calc_residual(triggers, lat_rad, lon_rad, depth_km):
    """Hitung total squared residual untuk satu kandidat hipocenter."""
    lat_deg = math.degrees(lat_rad)
    lon_deg = math.degrees(lon_rad)

    # Estimasi origin time dari rata-rata
    predicted_ots = []
    for tr in triggers:
        d_deg = haversine_deg(lat_deg, lon_deg, tr["lat"], tr["lon"])
        tt    = get_taup_time(d_deg, depth_km)
        predicted_ots.append(tr["t_arrive"] - tt)

    ot   = float(np.median(predicted_ots))
    sq_r = 0.0
    for i, tr in enumerate(triggers):
        d_deg = haversine_deg(lat_deg, lon_deg, tr["lat"], tr["lon"])
        tt    = get_taup_time(d_deg, depth_km)
        pred  = ot + tt
        sq_r += (pred - tr["t_arrive"]) ** 2

    return sq_r, ot

# Magnitudo & MMI
def estimate_magnitude(triggers, epicenter):
    mls = []
    for tr in triggers:
        if tr["peak_amp"] <= 0:
            continue
        d_km = dist_km(epicenter["lat"], epicenter["lon"], tr["lat"], tr["lon"])
        d_km = max(d_km, 1.0)
        delta = d_km / 111.19
        if delta <= 0:
            continue
        ml = math.log10(tr["peak_amp"]) + 3 * math.log10(8.0 * delta) - 2.92
        mls.append(ml)
    if not mls:
        return None
    return round(float(np.median(mls)), 1)

def estimate_mmi(mag, depth_km):
    """Wald et al. 1999 + atenuasi kedalaman."""
    if mag is None:
        return "I", "Tidak terasa"
    mmi_val = 1.5 * mag - 0.5 * math.log10(max(depth_km, 1)) - 1.0
    mmi_val = max(1, min(10, round(mmi_val)))
    mmi_map = {
        1: ("I",    "Tidak terasa"),
        2: ("II",   "Sangat lemah"),
        3: ("III",  "Lemah"),
        4: ("IV",   "Cukup terasa"),
        5: ("V",    "Kuat"),
        6: ("VI",   "Sangat kuat"),
        7: ("VII",  "Kerusakan ringan"),
        8: ("VIII", "Kerusakan sedang"),
        9: ("IX",   "Kerusakan berat"),
        10:("X",    "Kerusakan sangat berat"),
    }
    return mmi_map.get(mmi_val, ("I", "Tidak terasa"))

def alert_level(mag, n_sta):
    """GlobalQuake-style alert level 0-4."""
    level = 0
    for lvl in sorted(ALERT_LEVELS.keys(), reverse=True):
        cfg = ALERT_LEVELS[lvl]
        if mag is not None and mag >= cfg["min_mag"] and n_sta >= cfg["min_sta"]:
            level = lvl
            break
    return level

def potential_desc(mag, depth_km):
    if mag is None:
        return "Dalam analisis"
    if mag >= 7.0 and depth_km <= 70:
        return "Berpotensi tsunami"
    elif mag >= 6.5:
        return "Berpotensi merusak"
    elif mag >= 5.0:
        return "Dapat dirasakan luas"
    elif mag >= 4.0:
        return "Dapat dirasakan lokal"
    return "Umumnya tidak dirasakan"

#  SeedLink 
class EWSClient(EasySeedLinkClient):
    def on_data(self, trace):
        sta = trace.stats.station
        if sta not in sta_buffers:
            return

        with lock:
            buf = sta_buffers[sta]
            sr  = float(trace.stats.sampling_rate)
            buf["sr"] = sr

            new_maxlen = int(sr * 300)
            if buf["data"].maxlen != new_maxlen:
                buf["data"] = collections.deque(buf["data"], maxlen=new_maxlen)

            for v in trace.data:
                buf["data"].append(float(v))

            arr    = np.array(buf["data"])
            sr_int = int(sr)
            if len(arr) < sr_int * 15:
                return

            # STA/LTA
            cft    = classic_sta_lta(arr, int(STA_SEC*sr_int), int(LTA_SEC*sr_int))
            on_off = trigger_onset(cft, THR_ON, THR_OFF)

            if len(on_off) > 0 and not buf["triggered"]:
                buf["triggered"]    = True
                buf["trigger_time"] = time.time()
                
                buf["peak_amp"]     = float(np.abs(arr[-sr_int*10:]).max())
                print(f"[TRIGGER] {sta} t={time.strftime('%H:%M:%S')} amp={buf['peak_amp']:.0f}")

            elif len(on_off) == 0 and buf["triggered"]:
                if time.time() - buf["trigger_time"] > 120:
                    buf["triggered"]    = False
                    buf["trigger_time"] = None
                    buf["peak_amp"]     = 0.0

    def on_seedlink_error(self):
        print("SeedLink error...")

def run_seedlink():
    while True:
        try:
            client = EWSClient(f"{GEOFON_HOST}:{GEOFON_PORT}")
            for s in STATIONS:
                try:
                    client.select_stream(s["net"], s["sta"], s["cha"])
                    print(f"  Subscribed: {s['sta']}")
                except Exception as e:
                    print(f"  Skip {s['sta']}: {e}")
            print("SeedLink terhubung!")
            client.run()
        except Exception as e:
            print(f"SeedLink reconnect: {e}")
            time.sleep(10)

# Proses trigger 
def process_triggers():
    now = time.time()
    with lock:
        triggers = []
        for s in STATIONS:
            buf = sta_buffers[s["sta"]]
            if buf["triggered"] and buf["trigger_time"] is not None:
                if now - buf["trigger_time"] <= ASSOC_WINDOW:
                    triggers.append({
                        "sta"      : s["sta"],
                        "lat"      : s["lat"],
                        "lon"      : s["lon"],
                        "label"    : s["label"],
                        "t_arrive" : buf["trigger_time"],
                        "peak_amp" : buf["peak_amp"],
                    })

    if len(triggers) < MIN_STATIONS:
        return None

    # Key event =
    event_key = str(int(min(t["t_arrive"] for t in triggers) / 60))
    if event_key in active_events:
        return None

    print(f"[EWS] {len(triggers)} stasiun trigger — menghitung lokasi...")

    # Grid search
    epi = spiral_grid_search(triggers)
    if epi is None:
        return None

    # Filter: lokasi harus dalam area Indonesia
    if not (-15 <= epi["lat"] <= 10 and 90 <= epi["lon"] <= 145):
        print(f"[EWS] Lokasi di luar area ({epi['lat']:.1f}, {epi['lon']:.1f}) — abaikan")
        return None

    # Filter RMS 
    if epi["rms_sec"] > 20:
        print(f"[EWS] RMS terlalu besar ({epi['rms_sec']:.1f}s) — abaikan")
        return None

    mag   = estimate_magnitude(triggers, epi)
    mmi_code, mmi_desc = estimate_mmi(mag, epi["depth_km"])
    level = alert_level(mag if mag else 0, epi["n_sta"])
    kab_name, prov, kab_km, kab_dir = nearest_kabupaten(epi["lat"], epi["lon"])
    potential = potential_desc(mag, epi["depth_km"])

    lat_str = f"{abs(epi['lat']):.2f}°{'LS' if epi['lat'] < 0 else 'LU'}"
    lon_str = f"{abs(epi['lon']):.2f}°{'BT' if epi['lon'] > 0 else 'BB'}"

    # Wilayah:
    wilayah = f"{kab_km:.0f} km {kab_dir} {kab_name}, {prov}"

    event = {
        "id"          : event_key,
        "lat"         : epi["lat"],
        "lon"         : epi["lon"],
        "lat_str"     : lat_str,
        "lon_str"     : lon_str,
        "depth_km"    : epi["depth_km"],
        "magnitude"   : mag,
        "mmi"         : mmi_code,
        "mmi_desc"    : mmi_desc,
        "wilayah"     : wilayah,
        "kabupaten"   : kab_name,
        "provinsi"    : prov,
        "kab_dist_km" : kab_km,
        "kab_dir"     : kab_dir,
        "potential"   : potential,
        "alert_level" : level,
        "alert_label" : ALERT_LEVELS[level]["label"],
        "n_stations"  : epi["n_sta"],
        "stations"    : [t["sta"] for t in triggers],
        "rms_sec"     : epi["rms_sec"],
        "conf_km"     : epi["conf_km"],
        "origin_time" : epi["origin_t"],
        "timestamp"   : time.time(),
    }

    active_events[event_key] = event
    print(f"[EWS] GEMPA! M{mag} {wilayah}")
    print(f"       Koordinat: {lat_str}, {lon_str} | Kedalaman: {epi['depth_km']} km")
    print(f"       MMI: {mmi_code} ({mmi_desc}) | Alert: Level {level}")
    print(f"       RMS: {epi['rms_sec']}s | Conf: ±{epi['conf_km']}km | {epi['n_sta']} stasiun")
    return event

#  WebSocket 
async def trigger_processor():
    while True:
        event = process_triggers()
        if event and connected_ws:
            msg  = json.dumps({"type": "ews_alert", "data": event})
            dead = set()
            for ws in connected_ws.copy():
                try:
                    await ws.send(msg)
                except Exception:
                    dead.add(ws)
            connected_ws -= dead
        await asyncio.sleep(2)

async def handler(websocket):
    connected_ws.add(websocket)
    print(f"Client terhubung: {websocket.remote_address}")
    try:
        #status awal
        with lock:
            status = [{
                "sta"      : s["sta"],
                "label"    : s["label"],
                "lat"      : s["lat"],
                "lon"      : s["lon"],
                "triggered": sta_buffers[s["sta"]]["triggered"],
            } for s in STATIONS]
        await websocket.send(json.dumps({"type": "station_status", "data": status}))

        while True:
            with lock:
                sta_status = [{
                    "sta"      : s["sta"],
                    "triggered": sta_buffers[s["sta"]]["triggered"],
                    "amp"      : round(sta_buffers[s["sta"]]["peak_amp"], 1),
                } for s in STATIONS]

            triggered_count = sum(1 for s in sta_status if s["triggered"])
            await websocket.send(json.dumps({
                "type"     : "heartbeat",
                "stations" : sta_status,
                "triggered": triggered_count,
                "time"     : time.time(),
            }))
            await asyncio.sleep(5)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_ws.discard(websocket)
        print("Client disconnect")

async def main():
    print(f"Menunggu SeedLink data (20 detik)...")
    await asyncio.sleep(20)
    port = int(os.environ.get("PORT", 8766))
    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"EWS WebSocket port {port}")
        asyncio.create_task(trigger_processor())
        await asyncio.Future()

threading.Thread(target=run_seedlink, daemon=True).start()
asyncio.run(main())
