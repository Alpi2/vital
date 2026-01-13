import { Component } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive, CommonModule],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  sidebarCollapsed = false;
  showUserMenu = false;
  showLanguageMenu = false;
  currentTheme: 'light' | 'dark' = 'light';
  
  currentLanguage = {
    code: 'TR',
    name: 'Türkçe',
    flag: '🇹🇷'
  };
  
  languages = [
    { code: 'TR', name: 'Türkçe', flag: '🇹🇷' },
    { code: 'EN', name: 'English', flag: '🇬🇧' },
    { code: 'DE', name: 'Deutsch', flag: '🇩🇪' },
    { code: 'FR', name: 'Français', flag: '🇫🇷' },
    { code: 'ES', name: 'Español', flag: '🇪🇸' },
    { code: 'AR', name: 'العربية', flag: '🇸🇦' },
    { code: 'ZH', name: '中文', flag: '🇨🇳' },
    { code: 'RU', name: 'Русский', flag: '🇷🇺' },
    { code: 'JA', name: '日本語', flag: '🇯🇵' },
    { code: 'PT', name: 'Português', flag: '🇵🇹' }
  ];
  
  toggleSidebar() {
    this.sidebarCollapsed = !this.sidebarCollapsed;
  }
  
  toggleLanguageMenu() {
    this.showLanguageMenu = !this.showLanguageMenu;
    this.showUserMenu = false;
  }
  
  toggleUserMenu() {
    this.showUserMenu = !this.showUserMenu;
    this.showLanguageMenu = false;
  }
  
  selectLanguage(lang: any) {
    this.currentLanguage = lang;
    this.showLanguageMenu = false;
    console.log('Dil değiştirildi:', lang.name);
  }
  
  toggleTheme() {
    this.currentTheme = this.currentTheme === 'light' ? 'dark' : 'light';
    document.body.classList.toggle('dark-theme');
    console.log('Tema değiştirildi:', this.currentTheme);
  }
}
