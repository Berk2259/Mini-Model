import torch
import torch.nn as nn

# --- Tokenizer: harfleri numaralara çeviriyoruz ---
metin = "anaconda" # Modelin çalışacağı kelime
harfler = sorted(set(metin)) # Kelimedeki farklı harflerin alfabetik listesi: a, c, d, n, o
harf_numara = {h: i for i, h in enumerate(harfler)} # Harften numaraya sözlük
numara_harf = {i: h for h, i in harf_numara.items()}  # Numaradan harfe sözlük (tersi)
sayilar = [harf_numara[h] for h in metin]  # Kelimeyi numara listesine çevirir

# --- Veri: her satır bir soru ("önceki ve şimdiki harften sonra ne gelir?") ---
onceki = torch.tensor(sayilar[:-2]) # Önceki harfler (son 2 eleman atılmış)
simdiki = torch.tensor(sayilar[1:-1]) # Şimdiki harfler (ilk ve son eleman atılmış)
hedef = torch.tensor(sayilar[2:])  # Doğru cevaplar (ilk 2 eleman atılmış)

print(sayilar) # Kelimenin numara hâli
print(onceki) # Önceki harflerin numaraları
print(simdiki)  # Şimdiki harflerin numaraları
print(hedef) # Doğru cevapların numaraları


# --- Model: iki tablo, toplam 35 parametre ---
tablo = nn.Embedding(len(harfler), 2) # Kart tablosu: her harfe 2 sayılık kart (5 x 2 = 10 parametre)
cikis = nn.Linear(4, len(harfler)) # Puanlama katmanı: 4 sayı girer, her harf için 1 puan çıkar (25 parametre)

kart_onceki = tablo(onceki) # Önceki harflerin kartlarını tablodan çeker
kart_simdiki = tablo(simdiki) # Şimdiki harflerin kartlarını tablodan çeker
kartlar = torch.cat([kart_onceki, kart_simdiki], dim=1)  # İki kartı yan yana birleştirir (2 + 2 = 4 sayı)

skorlar = cikis(kartlar) # Her soru için 5 harfin puanını hesaplar (tahmin)
kayip = nn.functional.cross_entropy(skorlar, hedef)  # Tahmini doğru cevaplarla karşılaştırıp ceza puanı (loss) verir

print(kartlar.shape) # Kartların boyutu: [6, 4]
print(skorlar.shape) # Puanların boyutu: [6, 5]
print(kayip) # Eğitimden önceki loss


# --- Eğitim ---
optimizer = torch.optim.SGD(list(tablo.parameters()) + list(cikis.parameters()), lr=0.5) # Optimizer: 35 parametreyi güncelleyecek araç, lr güncelleme hızı

# 100 kere gradient hesabı yaparak eğitim yapıyor
for adim in range(100):
    kartlar = torch.cat([tablo(onceki), tablo(simdiki)], dim=1)
    skorlar = cikis(kartlar)
    kayip = nn.functional.cross_entropy(skorlar, hedef)
    optimizer.zero_grad()
    kayip.backward()
    optimizer.step()
    if adim % 10 == 0:
        print(adim, kayip.item())

# --- Modele soruyoruz ---
with torch.no_grad(): # Sadece soruyoruz, öğrenme (gradient hesabı) yok
    kartlar = torch.cat([tablo(onceki), tablo(simdiki)], dim=1) # Eğitilmiş tabloyla kartları çek ve birleştir
    tahmin = cikis(kartlar).argmax(dim=1) # Her satırda en yüksek puanlı harfin numarasını seç (modelin cevabı)

print(tahmin) # Modelin cevap numaraları
print("".join(numara_harf[i.item()] for i in tahmin)) # Modelin cevabını harfe çevirip yazdır
print("".join(numara_harf[i.item()] for i in hedef)) # Doğru cevabı harfe çevirip yazdır (karşılaştırmak için)