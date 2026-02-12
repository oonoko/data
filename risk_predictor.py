#!/usr/bin/env python3
"""
Dzud Risk Predictor - MVP
Хэрэглэгч координат + малын тоо оруулахад эрсдэл тооцоолно
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, List, Tuple

class DzudRiskPredictor:
    def __init__(self):
        """Initialize predictor with weather data and model"""
        # Load weather data
        self.weather_data = pd.read_csv('weather_omnogovi_monthly_clean.csv')
        
        # Load model (if exists)
        try:
            self.model = joblib.load('dzud_risk_model_advanced.pkl')
            self.scaler = joblib.load('scaler_advanced.pkl')
            self.has_model = True
        except:
            self.has_model = False
            print("⚠️  Model not found, using rule-based system")
        
        # Livestock vulnerability weights (эмзэг байдал)
        self.livestock_weights = {
            'sheep': 1.2,    # хонь - эмзэг
            'goat': 1.3,     # ямаа - хамгийн эмзэг
            'cattle': 0.9,   # үхэр - тэсвэртэй
            'horse': 0.8,    # адуу - тэсвэртэй
            'camel': 0.6     # тэмээ - хамгийн тэсвэртэй
        }
    
    def find_nearest_location(self, lat: float, lon: float) -> Dict:
        """Find nearest weather station"""
        # Calculate distance to all locations
        self.weather_data['distance'] = np.sqrt(
            (self.weather_data['lat'] - lat)**2 + 
            (self.weather_data['lon'] - lon)**2
        )
        
        # Get nearest location
        nearest = self.weather_data.loc[self.weather_data['distance'].idxmin()]
        
        return {
            'aimag': nearest['aimag'],
            'soum': nearest['soum'],
            'lat': nearest['lat'],
            'lon': nearest['lon'],
            'distance_km': nearest['distance'] * 111  # degrees to km
        }
    
    def get_current_weather(self, lat: float, lon: float, month: int = None) -> Dict:
        """Get weather features for location and month"""
        if month is None:
            month = datetime.now().month
        
        # Find nearest location
        location = self.find_nearest_location(lat, lon)
        
        # Get weather for this location and month
        weather = self.weather_data[
            (self.weather_data['soum'] == location['soum']) &
            (self.weather_data['month'] == month)
        ].sort_values('year', ascending=False).iloc[0]
        
        return {
            'location': location,
            'month': month,
            'year': int(weather['year']),
            'avg_temp': float(weather['avg_temp']),
            'min_temp': float(weather['min_temp']),
            'wind_speed': float(weather['wind_speed']),
            'snowfall_sum': float(weather['snowfall_sum']),
            'precip_sum': float(weather['precip_sum'])
        }
    
    def calculate_weather_risk(self, weather: Dict) -> Tuple[float, List[str]]:
        """Calculate weather-based risk score (0-100)
        Зуд зөвхөн өвлийн сарууд (11, 12, 1, 2, 3) дээр тооцоологдоно
        """
        # Зун (4-10 сар) - зуд байхгүй
        if weather['month'] not in [11, 12, 1, 2, 3]:
            return 0, ["Зун - зудын эрсдэл байхгүй"]
        
        score = 0
        reasons = []
        
        # Temperature risk
        if weather['min_temp'] < -25:
            score += 35
            reasons.append(f"Хамгийн бага температур маш бага ({weather['min_temp']:.1f}°C)")
        elif weather['min_temp'] < -20:
            score += 25
            reasons.append(f"Хамгийн бага температур бага ({weather['min_temp']:.1f}°C)")
        elif weather['min_temp'] < -15:
            score += 15
            reasons.append(f"Температур доогуур ({weather['min_temp']:.1f}°C)")
        
        # Wind risk
        if weather['wind_speed'] > 18:
            score += 25
            reasons.append(f"Салхи маш хүчтэй ({weather['wind_speed']:.1f} м/с)")
        elif weather['wind_speed'] > 15:
            score += 15
            reasons.append(f"Салхи хүчтэй ({weather['wind_speed']:.1f} м/с)")
        elif weather['wind_speed'] > 12:
            score += 10
            reasons.append(f"Салхи дунд зэрэг ({weather['wind_speed']:.1f} м/с)")
        
        # Snowfall risk
        if weather['snowfall_sum'] > 10:
            score += 20
            reasons.append(f"Их цас орсон ({weather['snowfall_sum']:.1f} мм)")
        elif weather['snowfall_sum'] > 5:
            score += 10
            reasons.append(f"Цас орсон ({weather['snowfall_sum']:.1f} мм)")
        
        # Precipitation deficit (drought)
        if weather['precip_sum'] < 5:
            score += 15
            reasons.append(f"Хур тунадас маш бага ({weather['precip_sum']:.1f} мм)")
        elif weather['precip_sum'] < 10:
            score += 8
            reasons.append(f"Хур тунадас бага ({weather['precip_sum']:.1f} мм)")
        
        # Cold index (wind chill)
        cold_index = weather['min_temp'] - (weather['wind_speed'] * 0.5)
        if cold_index < -30:
            score += 15
            reasons.append(f"Хүйтний индекс өндөр ({cold_index:.1f})")
        
        return min(score, 100), reasons
    
    def calculate_livestock_exposure(self, livestock: Dict) -> Tuple[float, int]:
        """Calculate livestock exposure score (0-100)"""
        total_count = 0
        weighted_sum = 0
        
        for animal_type, count in livestock.items():
            if count > 0 and animal_type in self.livestock_weights:
                total_count += count
                weighted_sum += count * self.livestock_weights[animal_type]
        
        if total_count == 0:
            return 0, 0
        
        # Normalize to 0-100 scale
        # Assume 1000 animals = 50 points baseline
        exposure_score = min((weighted_sum / 1000) * 50, 100)
        
        return exposure_score, total_count
    
    def calculate_final_risk(self, weather_risk: float, exposure_score: float) -> Dict:
        """Calculate final risk score and level"""
        # Weighted combination
        final_score = (weather_risk * 0.7) + (exposure_score * 0.3)
        
        # Determine risk level
        if final_score < 25:
            level = 0
            label = "Бага"
            color = "green"
        elif final_score < 50:
            level = 1
            label = "Дунд"
            color = "yellow"
        elif final_score < 75:
            level = 2
            label = "Өндөр"
            color = "orange"
        else:
            level = 3
            label = "Маш өндөр"
            color = "red"
        
        return {
            'score': round(final_score, 1),
            'level': level,
            'label': label,
            'color': color,
            'weather_risk': round(weather_risk, 1),
            'exposure_score': round(exposure_score, 1)
        }
    
    def get_recommendations(self, risk_level: int, livestock: Dict, weather: Dict) -> Dict:
        """Generate action recommendations by livestock type"""
        recommendations = {}
        
        # Sheep and Goats (хонь, ямаа)
        if livestock.get('sheep', 0) > 0 or livestock.get('goat', 0) > 0:
            if risk_level >= 2:  # High risk
                recommendations['sheep_goat'] = [
                    "🏠 Салхи, хүйтнээс хамгаалах байр бэлтгэх",
                    "🌾 Нэмэлт тэжээл нөөцлөх (өвс, тэжээл)",
                    "💧 Усны хангамж шалгах",
                    "👥 Сүрэг бүлэглэн хамгаалах"
                ]
            else:
                recommendations['sheep_goat'] = [
                    "✓ Өвөлжилтийн бэлтгэл хангалттай эсэх шалгах",
                    "✓ Тэжээлийн нөөц хангалттай байх"
                ]
        
        # Cattle (үхэр)
        if livestock.get('cattle', 0) > 0:
            if risk_level >= 2:
                recommendations['cattle'] = [
                    "🏠 Хашаа, байр бэлтгэх",
                    "💧 Ус, тэжээлийн нөөц нэмэгдүүлэх",
                    "🌡️ Дулаан хадгалах арга хэмжээ"
                ]
            else:
                recommendations['cattle'] = [
                    "✓ Хэвийн өвөлжилтийн бэлтгэл"
                ]
        
        # Horses (адуу)
        if livestock.get('horse', 0) > 0:
            if risk_level >= 2:
                recommendations['horse'] = [
                    "🏃 Нүүх боломжтой газар бэлтгэх",
                    "🌾 Тэжээлийн нөөц",
                    "💧 Усны эх үүсвэр"
                ]
            else:
                recommendations['horse'] = [
                    "✓ Хэвийн өвөлжилт"
                ]
        
        # Camels (тэмээ)
        if livestock.get('camel', 0) > 0:
            recommendations['camel'] = [
                "✓ Тэмээ хамгийн тэсвэртэй",
                "✓ Ердийн арчилгаа хангалттай"
            ]
        
        # General recommendations
        general = []
        if risk_level >= 3:
            general.append("🚨 АНХААРУУЛГА: Маш өндөр эрсдэл!")
            general.append("📍 Эрсдэл багатай газар руу нүүх боломжийг судлах")
        if risk_level >= 2:
            general.append("⚠️  Цаг агаарын мэдээг тогтмол хянах")
            general.append("📞 Орон нутгийн мал эмнэлэгтэй холбоо барих")
        
        recommendations['general'] = general
        
        return recommendations
    
    def predict(self, lat: float, lon: float, livestock: Dict, month: int = None) -> Dict:
        """Main prediction function"""
        # Get weather data
        weather = self.get_current_weather(lat, lon, month)
        
        # Calculate weather risk
        weather_risk, weather_reasons = self.calculate_weather_risk(weather)
        
        # Calculate livestock exposure
        exposure_score, total_livestock = self.calculate_livestock_exposure(livestock)
        
        # Calculate final risk
        risk = self.calculate_final_risk(weather_risk, exposure_score)
        
        # Get recommendations
        recommendations = self.get_recommendations(risk['level'], livestock, weather)
        
        # Compile result
        result = {
            'location': weather['location'],
            'weather': weather,
            'risk': risk,
            'livestock': {
                'total': total_livestock,
                'breakdown': livestock,
                'exposure_score': exposure_score
            },
            'top_reasons': weather_reasons[:3],  # Top 3
            'recommendations': recommendations,
            'confidence': 'дунд' if self.has_model else 'бага',
            'note': 'Энэ нь туршилтын тооцоолол юм. Бодит мэдээлэл дээр үндэслэнэ үү.'
        }
        
        return result


# Example usage
if __name__ == "__main__":
    predictor = DzudRiskPredictor()
    
    # Test case
    result = predictor.predict(
        lat=43.5,
        lon=104.4,
        livestock={
            'sheep': 200,
            'goat': 150,
            'cattle': 50,
            'horse': 30,
            'camel': 10
        },
        month=1  # January
    )
    
    print("="*60)
    print("ЗУДЫН ЭРСДЭЛИЙН ҮНЭЛГЭЭ")
    print("="*60)
    print(f"\n📍 Байршил: {result['location']['soum']}, {result['location']['aimag']}")
    print(f"   Координат: {result['location']['lat']:.2f}, {result['location']['lon']:.2f}")
    print(f"\n🌡️  Цаг агаар ({result['weather']['month']}-р сар, {result['weather']['year']}):")
    print(f"   Дундаж температур: {result['weather']['avg_temp']:.1f}°C")
    print(f"   Хамгийн бага температур: {result['weather']['min_temp']:.1f}°C")
    print(f"   Салхины хурд: {result['weather']['wind_speed']:.1f} м/с")
    print(f"   Цас: {result['weather']['snowfall_sum']:.1f} мм")
    print(f"   Хур тунадас: {result['weather']['precip_sum']:.1f} мм")
    print(f"\n🎯 ЭРСДЭЛИЙН ДҮН: {result['risk']['score']}/100")
    print(f"   Түвшин: {result['risk']['label']} ({result['risk']['color']})")
    print(f"   Итгэлцүүр: {result['confidence']}")
    print(f"\n📊 Дэлгэрэнгүй:")
    print(f"   Цаг агаарын эрсдэл: {result['risk']['weather_risk']}/100")
    print(f"   Малын өртөлт: {result['risk']['exposure_score']}/100")
    print(f"\n🐑 Малын тоо: {result['livestock']['total']} толгой")
    for animal, count in result['livestock']['breakdown'].items():
        if count > 0:
            print(f"   {animal}: {count}")
    print(f"\n⚠️  Гол шалтгаанууд:")
    for i, reason in enumerate(result['top_reasons'], 1):
        print(f"   {i}. {reason}")
    print(f"\n💡 Зөвлөмж:")
    for category, recs in result['recommendations'].items():
        if recs:
            print(f"\n   {category.upper()}:")
            for rec in recs:
                print(f"      {rec}")
    print(f"\n📝 {result['note']}")
    print("="*60)
