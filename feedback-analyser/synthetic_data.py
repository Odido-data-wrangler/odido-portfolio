
# Enhanced code to generate 1000 rows of realistic telecom customer feedback data
import random
import pandas as pd
import datetime

# ==============================================
# EXPANDED TEMPLATES FOR BEN MOBILE COMMENTS
# ==============================================
ben_mobile_comments = [
  # Positive comments about pricing
  "Ben is echt betaalbaar in deze dure tijd | Vroeger moeste we per tijds gebeuren betalen daarom ben ik tevreden bij ben maar ik heb de tijd nog wel om te verlengen en u herhaalt het toch ook. Groetjes {name}",
  "Ik ben super blij met de scherpe prijzen bij Ben! Als student moet ik op mijn geld letten en Ben past perfect binnen m'n budget.",
  "Al {n} jaar klant bij Ben, voornamelijk vanwege hun eerlijke prijsbeleid. Geen verborgen kosten of vreemde verhogingen achteraf.",
  "Na vergelijking van verschillende aanbieders bleek Ben verreweg de meest voordelige voor mijn verbruik. Dat is al {n} jaar zo.",
  "De nieuwe bundel van Ben is perfect voor mij, alle GB's die ik nodig heb voor een scherpe prijs. Goed werk!",
  "Ben blijft consistent in hun prijzen, ook na het eerste jaar. Niet zoals {provider} die na een jaar ineens 40% duurder werd.",
  
  # Positive comments about customer service
  "Al {n} jaar klant en erg tevreden over de klantenservice. Altijd snelle reacties en goede oplossingen bij problemen!",
  "Gister gebeld met een vraag over m'n verbruik en binnen 2 minuten had ik een medewerker aan de lijn die me perfect kon helpen!",
  "De medewerker bij Ben heeft echt met me meegedacht voor het beste abonnement. Niet pusherig voor de duurste optie zoals bij {provider}.",
  "Mijn simkaart was stuk. Belde Ben maandag en woensdag had ik al een nieuwe in huis. Top service!",
  "Klantenservice via chat werkt super goed, binnen 5 minuten probleem opgelost zonder eindeloos in de wacht te staan.",
  "Ben heeft me geholpen met het overzetten van mijn nummer van {provider}. Ging veel soepeler dan ik had verwacht.",
  
  # Positive comments about network quality
  "Overgestapt naar Ben vanwege de lage prijzen. Tot nu toe zeer tevreden met de kwaliteit van het netwerk.",
  "Ben gebruikt gewoon het netwerk van T-Mobile, dus de dekking is prima. Ik kom nergens waar ik geen bereik heb.",
  "Ook in de buitengebieden rond {location} heb ik met Ben altijd goed bereik, zelfs beter dan mijn collega's met {provider}.",
  "De 5G van Ben werkt uitstekend in mijn regio. Downloads zijn super snel en ik kan overal probleemloos videobellen.",
  "Als forens is netwerkkwaliteit belangrijk voor me. Na {n} maanden Ben kan ik zeggen dat ik nog nooit problemen heb gehad.",
  "Ook in drukke gebieden zoals op stations merk ik dat het netwerk van Ben prima blijft werken, geen vertragingen.",
  
  # Positive comments about product/app
  "Ben is een prima provider voor ons gezin. De familiekorting is fijn en het werkt allemaal zoals het moet.",
  "De Ben app is echt gebruiksvriendelijk. Makkelijk te zien hoeveel data ik nog over heb en wanneer mijn factuur komt.",
  "Makkelijk om via de app extra bundels toe te voegen als ik een keer meer nodig heb. Binnen een minuut geregeld!",
  "De nieuwe functie in de Ben app om je dataverbruik per app te zien vind ik super handig. Nu zie ik precies waar mijn data naartoe gaat.",
  "Online verlengen via de Ben website ging super soepel, overzichtelijke vergelijking van opties en direct geregeld.",
  "Fijn dat ik in de app precies kan zien wat ik in het buitenland heb verbruikt tijdens mijn vakantie in {location}.",
  
  # Mixed comments (some criticism)
  "Het onbeperkte data pakket is goed, maar ik vind het wel aan de prijzige kant vergeleken met vroeger.",
  "Gisteren gebeld met een probleem en werd snel geholpen. Alleen jammer dat ik eerst door een menu moest.",
  "Over het algemeen tevreden met Ben, maar zou wel wat meer kortingsacties willen zien voor bestaande klanten.",
  "De dekking is meestal goed maar in {location} heb ik regelmatig geen bereik, dat zou beter kunnen.",
  "Ben is prima qua prijs, maar de klantenservice is soms minder bereikbaar in de avonduren wanneer ik juist tijd heb om te bellen.",
  "Ik mis bij Ben wat extra diensten die {provider} wel biedt, zoals een streamingdienst inbegrepen bij je abonnement.",
  
  # Loyalty comments
  "De lente aanbieding was echt top! Zo veel data voor zo weinig geld, ben er erg blij mee.",
  "De sim only deal was precies wat ik zocht. Geen gedoe met toestel, gewoon een goede bundel.",
  "Als klant van {n} jaar blij met de loyaliteitsbonus. Mooi gebaar van Ben!",
  "Het verlengen ging soepel en de klantenservice gaf me goed advies over welk pakket het beste bij mijn verbruik past.",
  "Ben heeft me een mooi aanbod gedaan om te verlengen. Fijn dat trouwe klanten ook worden beloond en niet alleen nieuwe klanten.",
  "Na {n} jaar klant te zijn kreeg ik een leuke verrassing van Ben: 5GB extra data zonder meerkosten. Zo maak je klanten blij!",
  
  # Regional/specific comments
  "Ben heeft een uitstekende dekking, ook in landelijke gebieden waar andere providers het laten afweten.",
  "Ik ben overgestapt van {provider} naar Ben en merk geen verschil in kwaliteit, maar wel in prijs!",
  "De Ben app is overzichtelijk en werkt snel. Makkelijk om je verbruik te controleren.",
  "Na jaren klant te zijn geweest, blijf ik bij Ben vanwege de constante prijs-kwaliteit verhouding.",
  "De klantenservice wist mijn probleem met roaming tijdens mijn vakantie in {location} snel op te lossen.",
  "Ik woon in een nieuwbouwwijk in {location} en had met {provider} constant problemen, maar met Ben werkt alles perfect.",
  
  # Specific feature feedback
  "De optie om data te delen met familieleden vind ik super handig. Mijn dochter heeft nooit meer een lege bundel.",
  "Sinds de laatste update van de Ben app laadt hij veel sneller. Fijn dat jullie ook aan dat soort details denken.",
  "De facturering is altijd duidelijk en op tijd. Geen verrassingen, precies wat ik heb afgesproken.",
  "De mogelijkheid om tijdelijk extra data te kopen voor €5 heeft me echt gered toen ik een paar dagen extra moest werken op locatie.",
  "Fijn dat Ben ook eSIM ondersteunt. Mijn nieuwe telefoon heeft dual sim en het was heel makkelijk om in te stellen.",
  "De Ben bundel blijft geldig binnen de hele EU. Was laatst in {location} en kon probleemloos mijn data gebruiken.",
  
  # Detailed specific feedback
  "De nieuwe EU-roaming regels maken Ben nog aantrekkelijker. Was in {location} op vakantie en kon gewoon m'n normale bundel gebruiken zonder extra kosten.",
  "Als zzp'er is een betrouwbare provider cruciaal voor mijn werk. Ben heeft me nog nooit in de steek gelaten, ook niet op drukke locaties.",
  "Het overstapaanbod van Ben was niet te weerstaan. Kreeg €150 korting op een nieuw toestel én een voordeliger abonnement dan bij {provider}.",
  "De samenwerking tussen Ben en Spotify Premium scheelt me €10 per maand. Dat tikt aan op jaarbasis!",
  "Mijn dochter liet haar telefoon vallen en had direct een nieuwe simkaart nodig. Ben stuurde er gratis eentje op, volgende dag in huis.",
  "Ben blijft goed communiceren over wijzigingen. Kreeg keurig 2 maanden van tevoren bericht dat mijn contract afliep, met duidelijke opties."
]

# ==============================================
# EXPANDED TEMPLATES FOR ODIDO FIXED/MOBILE COMMENTS
# ==============================================
odido_fixed_comments = [
  # Severe technical issues
  "Mijn verbinding werkt dus niet! | Mijn verbinding werkt niet. Volgens de Guidion monteur die al meerdere malen hier is geweest is er sprake van een zogenoemde 'kruising'. Ik heb een abonnement bij jullie afgesloten in November 2024. En nu is nog steeds de storing niet opgelost door jullie",
  "Al {n} weken geen stabiel internet. Elke avond tussen 19:00 en 22:00 valt de verbinding steeds weg. Precies wanneer we willen streamen!",
  "Na de overstap van KPN naar Odido heb ik alleen maar problemen. Wifi bereikt de helft van het huis niet meer en de verbinding valt regelmatig weg.",
  "De glasvezel verbinding is nu voor de 3e keer in één week uitgevallen zonder duidelijke reden. Wanneer gaat dit eindelijk structureel opgelost worden?",
  "Internet werkt maar op 10% van de beloofde snelheid. Speedtests tonen 25Mbps terwijl ik betaal voor 250Mbps! Complete oplichterij.",
  "Sinds de overgang naar Odido is mijn upload snelheid dramatisch laag. Kan hierdoor niet meer thuiswerken bij videovergaderingen. Onacceptabel!",
  
  # TV box issues
  "De TV box hapert constant, bij elke zender. Heb al {n}x gebeld maar nog steeds niet opgelost. Nu weer 30 minuten in de wacht...",
  "Elke keer als het regent, valt het TV-signaal weg. De monteur zegt dat de bekabeling buiten niet goed is, maar niemand lost het op!",
  "De nieuwe Odido TV app op mijn smart TV crasht om de 15 minuten. Compleet onbruikbaar en een grote stap terug vergeleken met {provider}.",
  "Opnames van programma's verdwijnen soms zomaar van de TV box, zonder uitleg. Al 2x gemeld maar nog steeds geen oplossing.",
  "Het geluid van de TV box loopt niet synchroon met het beeld. Bij elk programma loopt het geluid ongeveer 2 seconden achter.",
  "De nieuwe update van de TV software heeft alle instellingen veranderd. Nu moet ik alles opnieuw instellen en werkt de helft van de apps niet meer.",
  
  # Installation/maintenance issues
  "Al {n} weken internetproblemen. Constant uitval in de avond. Monteur is langs geweest maar volgende dag weer hetzelfde probleem. Dit is echt beneden peil!",
  "3x vrij genomen voor de monteur en 3x niet komen opdagen zonder bericht. Dit kost me vrije dagen en nog steeds geen werkend internet!",
  "Monteur is langs geweest maar kon het probleem niet oplossen omdat hij 'de juiste apparatuur' niet bij zich had. Nu weer een week wachten op een nieuwe afspraak!",
  "Aansluiting van glasvezel zou 'eenvoudig' zijn volgens jullie. Maar nu zit ik met een half afgewerkte muur en kabels die uit het plafond hangen!",
  "Na het bezoek van de monteur werkt internet nog steeds niet, maar nu is ook mijn TV-aansluiting kapot. Hoe is dit mogelijk?",
  "De monteur heeft een tijdelijke oplossing gemaakt die na 2 dagen alweer niet meer werkt. Dus nu mag ik weer een week wachten op een nieuwe afspraak.",
  
  # Price/contract issues
  "De glasvezel is wel snel als het werkt, maar ik vind het duur voor wat je krijgt. En bij regen valt het signaal vaak weg.",
  "Zonder waarschuwing is mijn maandelijkse factuur verhoogd met €7,50. Bij navraag blijkt dit een 'inflatiecorrectie', maar wel 3x hoger dan de daadwerkelijke inflatie!",
  "Ik zou volgens jullie aanbieding de eerste 6 maanden maar €25 betalen, maar vanaf maand 1 wordt er €42,50 afgeschreven. Bij klantenservice krijg ik geen gehoor.",
  "In het contract staat dat ik na een jaar kan opzeggen, maar nu ik dat wil doen blijkt er ineens een opzegtermijn van 3 maanden te zijn. Dit is misleiding!",
  "De bundel zou 'onbeperkt' zijn, maar na 10GB krijg ik een sms dat mijn snelheid wordt verlaagd. Dat is toch niet wat 'onbeperkt' betekent?",
  "Bij het afsluiten werd gezegd dat installatie gratis was, maar nu zie ik €85 installatiekosten op mijn eerste factuur. Pure misleiding dus!",
  
  # Mixed (some positive) comments
  "De combinatie van vast internet en mobiel werkt prima bij Odido. Fijn dat alles op één factuur staat en goed overzichtelijk is in de app.",
  "De snelheid van de glasvezel is uitstekend, maar de TV dienst hapert regelmatig. Kan daar echt niets aan gedaan worden?",
  "Over het internet ben ik zeer tevreden, maar de klantenservice is dramatisch slecht. Ellenlange wachttijden en medewerkers die niet weten waar ze het over hebben.",
  "De TV app op mijn telefoon werkt perfect, maar op mijn smart TV crasht hij voortdurend. Vreemd contrast in kwaliteit van dezelfde app.",
  "Als je eenmaal door de installatieproblemen heen bent, is de dienst prima. Maar die eerste weken waren echt een hel.",
  "In vergelijking met {provider} is de TV-interface veel gebruiksvriendelijker, maar het aantal beschikbare zenders is minder. Beetje jammer.",
  
# Customer service complaints continued
  "Na de overgang naar Odido is alles slechter geworden. Storingen, lange wachttijden bij klantenservice, en de TV-app crasht voortdurend.",
  "Al 4x via verschillende kanalen een klacht ingediend, maar nooit een reactie ontvangen. Is er überhaupt nog een klantenservice?",
  "45 minuten in de wacht gestaan bij de klantenservice om uiteindelijk een medewerker te spreken die me niet kon helpen. Wat een tijdverspilling!",
  "De klantenservice van Odido is echt dramatisch vergeleken met {provider}. Niemand die zijn verantwoordelijkheid neemt voor de problemen.",
  "Chat, telefoon, e-mail geprobeerd - niemand kan me vertellen wanneer de monteur nu eindelijk komt om mijn internet te fixen. Al 3 weken offline!",
  "Ongelofelijk hoe ongeïnteresseerd de klantenservice klinkt. Ik ben al 10 jaar klant maar wordt behandeld alsof ik onbelangrijk ben.",
  "Elke keer als ik bel krijg ik een ander verhaal te horen. De ene medewerker zegt dit, de andere dat. Niemand die de geschiedenis van mijn klacht kent.",
  "Escalatie aangevraagd maar daar hoor je ook niks meer van. Je wordt gewoon aan het lijntje gehouden tot je opgeeft.",
  "De chatbot begrijpt mijn probleem niet en verwijst me steeds naar irrelevante FAQ pagina's. Als ik dan eindelijk een mens spreek, moet ik weer opnieuw beginnen.",
  "De klantenservice is alleen bereikbaar tijdens kantooruren. Dus als werkende moet ik vrij nemen om mijn internetproblemen op te lossen. Absurd.",
  
  # Wifi/Router issues
  "Helaas werkt mijn wifi regelmatig niet. Na contact met de helpdesk kreeg ik te horen dat er een storing in de wijk is. Dit duurt nu al {n} weken! Wanneer wordt dit opgelost?",
  "De nieuwe router die ik van Odido heb gekregen heeft veel minder bereik dan mijn oude router van {provider}. Nu heb ik in mijn slaapkamer geen wifi meer.",
  "Wifi valt steeds weg bij meer dan 5 apparaten tegelijk. Met thuiswerken en thuisonderwijs is dat echt een groot probleem voor ons gezin.",
  "Router start zichzelf om de paar uur opnieuw op zonder reden. Alle apparaten moeten dan opnieuw verbinding maken. Zeer hinderlijk tijdens online vergaderingen!",
  "De wifi-snelheid is veel lager dan beloofd. Naast de router meet ik 30Mbps terwijl ik voor 300Mbps betaal. Op andere plekken in huis is het nog veel slechter.",
  "Na de 'upgrade' naar de nieuwe Odido router is mijn wifi bereik gehalveerd. Nu moet ik extra repeaters kopen terwijl dat eerst niet nodig was.",
  
  # Weather-related issues
  "Bij elke regenbui valt mijn internet uit. Drie monteurs verder en nog steeds niet opgelost. Onacceptabel in 2025! Ik betaal wel elke maand de volle prijs.",
  "Zodra het gaat stormen, wordt mijn internetverbinding instabiel. Blijkbaar is de infrastructuur niet bestand tegen wat wind en regen.",
  "Als het warmer is dan 25 graden, begint mijn modem rare dingen te doen. Oververhitting? Vreemd dat dit met mijn oude {provider} modem nooit een probleem was.",
  "In de winter valt de verbinding vaker uit dan in de zomer. Niemand kan me uitleggen waarom dat zo is, maar het is een patroon dat al jaren terug gaat.",
  "Elke donderdagavond tussen 19:00 en 21:00 is mijn internet trager dan normaal. Zouden meer mensen in de wijk dan online zijn? Bij {provider} had ik dit probleem niet.",
  "Als het mistig is, is mijn TV-signaal opeens veel slechter. Monteur zegt dat dit niet kan, maar het is een patroon dat ik al maanden zie.",
  
  # Specific technical issues
  "Mijn IPv6 verbinding werkt niet goed waardoor ik problemen heb met bepaalde online diensten. Support kan me hier niet mee helpen omdat ze niet weten wat IPv6 is!",
  "Port forwarding instellen via de Odido router is onmogelijk. De handleiding klopt niet met wat ik in de interface zie en support kan niet helpen.",
  "De DNS servers van Odido zijn traag en onbetrouwbaar. Ik moet handmatig andere DNS servers instellen om fatsoenlijk te kunnen internetten.",
  "VPN verbindingen vallen steeds weg sinds ik ben overgestapt naar Odido. Voor mijn werk is dit een ramp en niemand kan me helpen.",
  "Kan mijn slimme thermostaat niet verbinden met de nieuwe router. Bij {provider} werkte dit probleemloos, nu zit ik in de kou.",
  "De latency (ping) op mijn verbinding fluctueert enorm. Online gamen is daardoor onmogelijk geworden sinds ik ben overgestapt naar Odido."
]

# Define business units
business_units = ["B2C - BEN - MOBILE", "B2C - Odido_mobile_fixed"]

# Define campaign types
campaign_types = ["Unknown", "Retention", "Acquisition", "Loyalty", "Family", "Data", "Cross-sell"]
campaign_names = {
  "Retention": ["Verlenging 2025", "Klantbehoud", "Zomercampagne", "Triple Play", "Internet & TV", "Voorjaarsverlenging"],
  "Acquisition": ["Nieuwe Klanten", "Voorjaar 2025", "Glasvezel 2025", "Sim Only Deal", "Welkom Deal"],
  "Loyalty": ["Loyaliteit 2025", "Klant Beloning", "Loyaliteit Plus", "Trouwe Klanten"],
  "Family": ["Familie bundel", "Gezinspakket", "Familie Deal"],
  "Data": ["Unlimited Data", "Extra Data", "Onbeperkt Data"],
  "Cross-sell": ["TV & Internet", "4K TV", "TV Pakket", "Mobile Add-on"]
}

# Define reasons
reasons = ["Brand", "Price", "Service", "Product"]

# Define names for personalization
names = ["De Vries", "Jansen", "De Jong", "Bakker", "Visser", "De Boer", "Mulder", "De Groot", "Bos", "Vos", 
         "Van Dijk", "Meijer", "Van der Berg", "Hendriks", "Dekker", "Smit", "De Wit", "Dijkstra", "Smits", "Kok"]

# Define other providers for reference
other_providers = ["KPN", "Ziggo", "Vodafone", "T-Mobile", "Tele2", "XS4ALL"]

# Define locations
locations = ["Amsterdam", "Rotterdam", "Utrecht", "Den Haag", "Eindhoven", "Groningen", "Tilburg", "Almere", 
             "Breda", "Nijmegen", "Enschede", "Apeldoorn", "Haarlem", "Amersfoort", "Zaanstad", "Arnhem", 
             "Spanje", "Duitsland", "Frankrijk", "Italië", "Griekenland"]

# Function to generate a random date within a range
def random_date(start, end):
    time_between_dates = end - start
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start + datetime.timedelta(days=random_number_of_days)
    return random_date.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

# Function to replace templates in comments
def personalize_comment(comment):
    comment = comment.replace("{name}", random.choice(names))
    comment = comment.replace("{n}", str(random.randint(1, 10)))
    comment = comment.replace("{provider}", random.choice(other_providers))
    comment = comment.replace("{location}", random.choice(locations))
    return comment

# Generate 1000 rows of data
rows = []
start_date = datetime.datetime(2025, 1, 1)
end_date = datetime.datetime(2025, 3, 25)
list_id = 4000
campaign_id = 10900
mail_id = 2300

for i in range(1000):
    business_unit = random.choice(business_units)
    campaign_type = random.choice(campaign_types)
    
    campaign_name = ""
    mail_name = ""
    
    if campaign_type != "Unknown":
        campaign_name = random.choice(campaign_names.get(campaign_type, []))
        mail_name = campaign_name
    
    reason = random.choice(reasons)
    
    # Assign appropriate score based on business unit
    if business_unit == "B2C - BEN - MOBILE":
        # BEN scores tend to be higher (6-8)
        score = random.randint(6, 8)
    else:
        # Odido scores tend to be more varied but lower (2-8)
        if random.random() < 0.6:
            # 60% chance of lower score for Odido
            score = random.randint(2, 4)
        else:
            # 40% chance of higher score
            score = random.randint(6, 8)
    
    # Generate appropriate comment
    if business_unit == "B2C - BEN - MOBILE":
        comment = personalize_comment(random.choice(ben_mobile_comments))
    else:
        comment = personalize_comment(random.choice(odido_fixed_comments))
    
    # Generate timestamp
    timestamp = random_date(start_date, end_date)
    
    rows.append({
        "Business Unit": business_unit,
        "List Id": list_id,
        "Campaign Id": campaign_id,
        "Campaign Name": campaign_name,
        "Campaign Type": campaign_type,
        "Mail Id": mail_id if campaign_type != "Unknown" else "",
        "Mail Name": mail_name,
        "Reason": reason,
        "Score": score,
        "Comment": comment,
        "Event Timestamp": timestamp
    })
    
    # Increment IDs for next row
    list_id += 1
    campaign_id += 1
    
    if campaign_type != "Unknown":
        mail_id += 1

# Create DataFrame and export to Excel
df = pd.DataFrame(rows)
df.to_excel("Synthetic_Data_and_AI_results.xlsx", index=False)

print(f"Generated 1000 rows of synthetic data. File saved as Synthetic_Data_and_AI_results.xlsx")