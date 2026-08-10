import React, { useContext } from 'react';
import { Link } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';
import { LanguageContext } from '../context/LanguageContext';
import { User, LogOut } from 'lucide-react';

const Navbar = () => {
  const { user, logout } = useContext(AuthContext);
  const { lang, toggleLanguage, t } = useContext(LanguageContext);

  return (
    <nav className="bg-white border-b-2 border-black sticky top-0 z-50 shadow-sm">
      <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-20">
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center mr-10">
              <Link to="/" className="font-serif font-black text-4xl tracking-tighter text-black uppercase">
                Jagrit<span className="text-red-600">.</span>
              </Link>
            </div>
            <div className="hidden sm:flex sm:space-x-8 h-full items-center">
              <Link to="/" className="border-transparent text-gray-500 hover:text-black hover:border-black inline-flex items-center px-1 pt-1 border-b-4 text-sm font-bold uppercase tracking-wider transition-colors h-full">
                {t('Top Stories')}
              </Link>
              <Link to="/foryou" className="border-transparent text-gray-500 hover:text-black hover:border-black inline-flex items-center px-1 pt-1 border-b-4 text-sm font-bold uppercase tracking-wider transition-colors h-full">
                {t('For You')}
              </Link>
              {user && (
                <Link to="/saved" className="border-transparent text-gray-500 hover:text-black hover:border-black inline-flex items-center px-1 pt-1 border-b-4 text-sm font-bold uppercase tracking-wider transition-colors h-full">
                  {t('Saved News')}
                </Link>
              )}
            </div>
          </div>
          
          <div className="flex items-center space-x-6">
            <button 
              onClick={toggleLanguage}
              className="px-4 py-1.5 border border-gray-300 text-gray-800 text-xs font-bold uppercase hover:bg-black hover:text-white transition-colors"
            >
              {lang === 'en' ? 'A/अ (HI)' : 'A/अ (EN)'}
            </button>
            
            {user ? (
              <div className="flex items-center space-x-4">
                <Link to="/preferences" className="text-gray-500 hover:text-black transition-colors" title={t('Preferences')}>
                  <User size={22} strokeWidth={1.5} />
                </Link>
                <button onClick={logout} className="text-gray-500 hover:text-black transition-colors" title={t('Logout')}>
                  <LogOut size={22} strokeWidth={1.5} />
                </button>
              </div>
            ) : (
              <Link to="/login" className="text-sm font-bold text-gray-500 hover:text-black uppercase tracking-wider transition-colors">
                {t('Login')}
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
